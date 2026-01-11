import re
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, TypedDict, Unpack

import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc
from sidecar_search.utils.gpu_utils import imap

from ..args import IndexSharedArgsMixin
from ..utils.datasets_utils import iter_tensors
from ..utils.faiss_utils import to_cpu, to_gpu
from ..provisioner import Provisioner

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths


# TODO: make a mixin for source, distinct from clean command
@dataclass
class IndexTrainArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["train"]]
):
    source: Path
    dimensions: int | None  # matryoshka
    normalize: bool  # also if normalize_d_
    preprocess: str
    clusters: int | None

    # not args
    ivf_encoding: str = field(init=False, compare=False)
    encoding_width: int = field(init=False, compare=False)

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("source", type=Path)
        parser.add_argument("-d", "--dimensions", default=None, type=int)
        parser.add_argument("-N", "--normalize", action="store_true")
        parser.add_argument("-p", "--preprocess", default="OPQ96_384")
        parser.add_argument("-c", "--clusters", default=None, type=int)

    def __post_init__(self):
        super().__post_init__()

        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')

        if self.dimensions is not None and not self.normalize:
            self.normalize = True
            warnings.warn("inferring --normalize from --dimension")

        if match := OPQ_PATTERN.match(self.preprocess):
            self.ivf_encoding = f"PQ{match[1]}"
            self.encoding_width = int(match[1])
            if self.encoding_width not in GPU_OPQ_WIDTHS:
                raise ValueError(f"OPQ width {self.encoding_width} is not valid")
        elif match := RR_PATTERN.match(self.preprocess):
            self.ivf_encoding = "SQ8"
            self.encoding_width = int(match[1])
        else:
            raise ValueError(f'preprocessing string "{self.preprocess}" is not valid')


class MemmapKwargs(TypedDict):
    dataset: Dataset
    shape: tuple[int, int]
    normalize: bool


type NDMemmap[T: np.generic] = np.memmap[Any, np.dtype[T]]


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

    @classmethod
    def with_train_args(cls, dataset: Dataset, args: IndexTrainArgs) -> Self:
        n = len(dataset)
        d = args.dimensions
        if d is None:
            d = len(dataset[0]["embedding"])
        return cls(dataset=dataset, shape=(n, d), normalize=args.normalize)

    def provision(self, progress: bool = False) -> NDMemmap[np.float32]:
        cache_path = self._compute_cache_path()
        if cache_path.exists():
            return np.memmap(cache_path, np.float32, mode="r", shape=self._shape)
        builder = MemmapBuilder(**self._kwargs)
        memmap = builder.build(cache_path, progress=progress)
        return memmap

    def _compute_cache_filename(self) -> str:
        return f"train_{self._compute_cache_hash()}.memmap"


def train_index(
    train: Dataset, factory_string: str, args: IndexTrainArgs
) -> faiss.Index:
    provisioner = MemmapProvisioner.with_train_args(train, args)
    train_memmap = provisioner.provision(progress=args.progress)

    # doing a bit of testing seems to show that passing METRIC_L2 is superior to passing
    # METRIC_INNER_PRODUCT for the same factory string, even for normalized embeddings
    _, d = train_memmap.shape
    index: faiss.Index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)

    index = to_gpu(index)
    index.train(train_memmap)  # type: ignore (monkey-patched)
    index = to_cpu(index)

    return index


def ensure_trained(dataset: Dataset, args: IndexTrainArgs):
    if args.clusters is None:
        clusters = len(dataset) // TRAIN_SIZE_MULTIPLE
    else:
        clusters = args.clusters
    factory_string = f"{args.preprocess},IVF{clusters},{args.ivf_encoding}"
    train_size = TRAIN_SIZE_MULTIPLE * clusters

    shuffled = dataset.shuffle(seed=42)
    train = shuffled.take(train_size)

    index = train_index(train, factory_string, args)
    with del_on_exc([args.empty_index_path, args.untuned_params_path]):
        faiss.write_index(index, str(args.empty_index_path))
        save_params(args.untuned_params_path, args.dimensions, args.normalize, None)
