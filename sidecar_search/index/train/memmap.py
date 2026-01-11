from pathlib import Path
from typing import Any, TypedDict, Unpack

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from sidecar_search.utils.contextmanager_utils import del_on_exc
from sidecar_search.utils.gpu_utils import imap

from ..provisioner import Provisioner
from ..utils.datasets_utils import iter_tensors

type NDMemmap[T: np.generic] = np.memmap[Any, np.dtype[T]]


class MemmapKwargs(TypedDict):
    dataset: Dataset
    shape: tuple[int, int]
    normalize: bool


class MemmapBuilder:
    def __init__(self, **kwargs: Unpack[MemmapKwargs]) -> None:
        self._dataset = kwargs["dataset"]
        self._shape = kwargs["shape"]
        self._normalize = kwargs["normalize"]

    def build(self, path: Path, progress: bool = False) -> NDMemmap[np.float32]:
        n, _ = self._shape
        i = 0
        memmap = np.memmap(path, np.float32, mode="w+", shape=self._shape)
        with (
            del_on_exc(path),
            tqdm(desc="create_memmap", total=n, disable=(not progress)) as counter,
        ):
            # save batches to disk by assigning to memmap slices
            batches = iter_tensors(self._dataset)
            batches = imap(batches, self._preproc, -1)
            for embeddings_batch in batches:
                n_batch = len(embeddings_batch)

                if i + n_batch > n:
                    n_batch = n - i
                    embeddings_batch = embeddings_batch[:n_batch]

                memmap[i : (i + n_batch)] = embeddings_batch.numpy()
                i += n_batch
                counter.update(n_batch)

                if i >= n:
                    break

            # memmap holds (now) leaked memory, so flush...
            memmap.flush()

        # ... and recreate memmap, letting the original go out-of-scope
        return np.memmap(path, np.float32, mode="r", shape=self._shape)

    def _preproc(self, _, embeddings: torch.Tensor) -> torch.Tensor:
        _, d = self._shape
        embeddings = embeddings[:, :d]
        if embeddings.shape[1] != d:
            raise ValueError("embeddings dimensions was less than d")
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings


class MemmapProvisioner(Provisioner[NDMemmap[np.float32]]):
    def __init__(self, **kwargs: Unpack[MemmapKwargs]) -> None:
        super().__init__(**kwargs)
        self._shape = kwargs["shape"]

    def provision(self, progress: bool = False) -> NDMemmap[np.float32]:
        cache_path = self._compute_cache_path()
        if cache_path.exists():
            return np.memmap(cache_path, np.float32, mode="r", shape=self._shape)
        builder = MemmapBuilder(**self._kwargs)
        memmap = builder.build(cache_path, progress=progress)
        return memmap

    def _compute_cache_filename(self) -> str:
        return f"train_{self._compute_cache_hash()}.memmap"
