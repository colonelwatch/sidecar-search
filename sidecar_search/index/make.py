from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Unpack

import faiss
import torch
from datasets import Dataset
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm

from sidecar_search.utils.contextmanager_utils import del_on_exc
from sidecar_search.utils.gpu_utils import imap, imap_multi_gpu

from .provisioner import Provisioner
from .utils.datasets_utils import iter_tensors
from .utils.faiss_utils import to_cpu, to_gpu

SHARD_SIZE = 1048576  # keep temporary shard sizes small to save on RAM


class MakeIndexKwargs(TypedDict):
    empty_index_path: Path
    dataset: Dataset
    holdouts: torch.Tensor | None
    d: int
    normalize: bool


@dataclass
class MakeIndexOutput:
    dir: Path

    @property
    def index_path(self) -> Path:
        return self.dir / "index.faiss"

    @property
    def ondisk_path(self) -> Path:
        return self.dir / "ondisk.ivfdata"

    def open(self) -> faiss.Index:
        # without IO_FLAG_ONDISK_SAME_DIR, read_index loads from working dir
        return faiss.read_index(str(self.index_path), faiss.IO_FLAG_ONDISK_SAME_DIR)


class MakeIndexBuilder:
    def __init__(self, **kwargs: Unpack[MakeIndexKwargs]) -> None:
        self._dataset = kwargs["dataset"]
        self._holdouts = kwargs["holdouts"]
        self._d = kwargs["d"]
        self._normalize = kwargs["normalize"]

        index = faiss.read_index(str(kwargs["empty_index_path"]))

        self._index = index
        self._on_gpus = [to_gpu(index, i) for i in range(torch.cuda.device_count())]

    def build(self, dir: Path, progress: bool = False) -> MakeIndexOutput:
        output_paths = MakeIndexOutput(dir)
        shard_paths: list[Path] = []
        try:
            n_full = len(self._dataset)
            n = n_full if self._holdouts is None else n_full - len(self._holdouts)
            with (
                tqdm(desc="make_index", total=n, disable=(not progress)) as c,
                ThreadPoolExecutor() as executor,
            ):
                for i_shard, row_start in enumerate(range(0, n_full, SHARD_SIZE)):
                    shard = self._dataset.select(  # yields another Datset not rows
                        range(row_start, min(row_start + SHARD_SIZE, n_full))
                    )

                    batches = iter_tensors(shard)
                    batches = imap(batches, self._preproc, -1)
                    counts = imap_multi_gpu(batches, self._add_with_gpu)
                    for count in counts:
                        c.update(count)

                    # transfer takes time, so do this across all GPUs in parallel
                    for on_cpu in executor.map(self._transfer_and_reset, self._on_gpus):
                        self._index.merge_from(on_cpu)

                    shard_path = dir / f"shard_{i_shard:03d}.faiss"
                    faiss.write_index(self._index, str(shard_path))
                    shard_paths.append(shard_path)

                    self._index.reset()

            # merge_ondisk takes file _names_ and only puts in working directory...
            temp_ondisk_path = Path("./ondisk.ivfdata")
            ondisk_path = output_paths.ondisk_path
            with del_on_exc([temp_ondisk_path, ondisk_path]):
                merge_ondisk(
                    self._index, [str(p) for p in shard_paths], temp_ondisk_path.name
                )
                temp_ondisk_path.rename(ondisk_path)  # ... so move it after
        finally:
            # drop the shards
            for p in shard_paths:
                p.unlink()

        # write the index (which points to `ondisk.ivfdata`)
        index_path = output_paths.index_path
        with del_on_exc(index_path):
            faiss.write_index(self._index, str(index_path))

        return output_paths

    def _preproc(
        self, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._holdouts is not None:
            not_in = torch.isin(ids, self._holdouts, invert=True)
            ids = ids[not_in]
            embeddings = embeddings[not_in]
        embeddings = embeddings[:, : self._d]
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings)
        return ids, embeddings

    def _add_with_gpu(
        self, device: torch.device, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> int:
        on_gpu = self._on_gpus[device.index]
        on_gpu.add_with_ids(  # type: ignore # faiss class_wrappers.py
            embeddings.numpy(), ids.numpy()
        )
        return len(ids)  # yield the number of embeddings added

    def _transfer_and_reset(self, on_gpu: faiss.Index) -> faiss.Index:
        on_cpu = to_cpu(on_gpu)
        on_gpu.reset()
        return on_cpu


class MakeIndexProvisioner(Provisioner[MakeIndexOutput]):
    def __init__(self, **kwargs: Unpack[MakeIndexKwargs]) -> None:
        super().__init__(**kwargs)

    def provision(self, progress: bool = False) -> MakeIndexOutput:
        cache_path = self._compute_cache_path()
        if cache_path.exists():
            return MakeIndexOutput(dir=cache_path)

        builder = MakeIndexBuilder(**self._kwargs)
        with del_on_exc(cache_path):
            cache_path.mkdir()
            out = builder.build(cache_path, progress=progress)

        return out

    def _compute_cache_filename(self) -> str:
        return f"make_index_{self._compute_cache_hash()}"
