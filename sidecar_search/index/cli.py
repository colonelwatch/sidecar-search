import json
import logging
import re
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import accumulate, tee
from pathlib import Path
from shutil import copy, rmtree
from typing import (
    Any,
    Generator,
    Iterable,
    Literal,
    Self,
    TypedDict,
    Unpack,
    assert_never,
    cast,
    get_args,
)

import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset, disable_progress_bars
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandGroupArgsBase, SubcommandArgsBase
from sidecar_search.utils.cache_utils import (
    clean_hf_cache,
    clean_persistent_cache,
    seal_hf_cache,
    seal_persistent_cache,
)
from sidecar_search.utils.gpu_utils import imap, imap_multi_gpu, iunsqueeze

from .provisioner import Provisioner

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths
BATCH_SIZE = 1024
SHARD_SIZE = 1048576  # keep temporary shard sizes small to save on RAM

logger = logging.getLogger(__name__)


class IndexParameters(TypedDict):
    recall: float  # in this case 10-recall@10
    exec_time: float  # seconds (raw faiss measure is in milliseconds)
    param_string: str  # pass directly to faiss index


class Params(TypedDict):
    dimensions: int | None
    normalize: bool
    optimal_params: list[IndexParameters] | None


@dataclass
class IndexSharedArgsMixin(SharedArgsMixin):
    build_dir: Path
    use_cache: bool  # for experiments only

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-B", "--build-dir", default=Path("."), type=Path)
        parser.add_argument("--use-cache", action="store_true")

    # TODO: rethink __post_init__ inheritance
    def __post_init__(self) -> None:
        if self.build_dir.exists() and not self.build_dir.is_dir():
            raise ValueError(
                f'build dir "{self.build_dir}" exists but is not a directory'
            )

    @property
    def empty_index_path(self) -> Path:
        return self.build_dir / "empty.faiss"

    @property
    def untuned_params_path(self) -> Path:
        return self.build_dir / "untuned.json"

    @property
    def params_path(self) -> Path:
        return self.build_dir / "params.json"

    @property
    def ids_path(self) -> Path:
        return self.build_dir / "ids.parquet"

    @property
    def index_paths(self) -> tuple[Path, Path]:
        return self.build_dir / "index.faiss", self.build_dir / "ondisk.ivfdata"


@dataclass
class IndexCleanArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["clean"]]
):
    source: Path | None

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-s", "--source", default=None, type=Path)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source is not None and not self.source.exists():
            raise ValueError(f'source at "{self.source}" does not exist')


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


@dataclass
class IndexTuneArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["tune"]]
):
    source: Path
    intersection: int | None  # 1R@1 else kR@k
    queries: int

    # not args
    one_recall_at_one: bool = field(init=False, compare=False)
    k: int = field(init=False, compare=False)
    dimensions: int | None = field(init=False, compare=False)
    normalize: bool = field(init=False, compare=False)

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-k", "--intersection", default=None, type=int)
        parser.add_argument("-q", "--queries", default=8192, type=int)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')
        if not self.empty_index_path.exists():
            raise ValueError(f'empty index "{self.empty_index_path}" does not exist')
        if not self.untuned_params_path.exists():
            raise ValueError(
                f'untuned params "{self.untuned_params_path}" does not exist'
            )

        if self.intersection is None:
            self.one_recall_at_one = True
            self.k = 1
        else:
            self.one_recall_at_one = False
            self.k = self.intersection

        with open(self.untuned_params_path) as f:
            params: Params = json.load(f)
        self.dimensions = params["dimensions"]
        self.normalize = params["normalize"]


@dataclass
class IndexFillArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["fill"]]
):
    source: Path

    # not args
    dimensions: int | None = field(init=False, compare=False)
    normalize: bool = field(init=False, compare=False)

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("source", type=Path)

    def __post_init__(self):
        super().__post_init__()

        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')
        if not self.empty_index_path.exists():
            raise ValueError(f'empty index "{self.empty_index_path}" does not exist')
        if not self.untuned_params_path.exists():
            raise ValueError(
                f'untuned params "{self.untuned_params_path}" does not exist'
            )

        with open(self.untuned_params_path) as f:
            params: Params = json.load(f)
        self.dimensions = params["dimensions"]
        self.normalize = params["normalize"]


AllIndexSubcommandArgs = IndexCleanArgs | IndexTrainArgs | IndexTuneArgs | IndexFillArgs

ALL_INDEX_SUBCOMMAND_ARGS = cast(
    tuple[type[AllIndexSubcommandArgs], ...], get_args(AllIndexSubcommandArgs)
)


@dataclass
class IndexGroupArgs(IndexSharedArgsMixin, CommandGroupArgsBase[Literal["index"]]):
    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        cls._add_subcommands(parser, ALL_INDEX_SUBCOMMAND_ARGS)


@contextmanager
def del_on_exc(path: Path | Iterable[Path]) -> Generator[None, None, None]:
    paths = [path] if isinstance(path, Path) else path
    try:
        yield
    except (KeyboardInterrupt, Exception):
        for p in paths:
            if not p.exists():
                continue
            if p.is_dir():
                rmtree(p)
            else:
                p.unlink()
        raise


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore  (wrong func signature)


def iter_tensors(
    dataset: Dataset,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    with dataset.formatted_as("torch", columns=["index", "embedding"]):
        for batch in dataset.iter(BATCH_SIZE):
            yield batch["index"], batch["embedding"]  # type: ignore


class GroundTruthKwargs(TypedDict):
    dataset: Dataset
    queries: Dataset
    do_inner_product_search: bool
    k: int


class GroundTruthBuilder:
    def __init__(self, **kwargs: Unpack[GroundTruthKwargs]) -> None:
        dataset = kwargs["dataset"]
        queries = kwargs["queries"]
        do_inner_product_search = kwargs["do_inner_product_search"]
        k = kwargs["k"]

        self._dataset = dataset
        self._queries = queries
        self._do_inner_product_search = do_inner_product_search
        self._k = k

        # get query embeddings and IDs, with a local copy for each GPU
        with queries.formatted_as("torch", columns=["embedding", "index"]):
            q_embeddings: torch.Tensor = queries["embedding"]  # type: ignore
            q_ids: torch.Tensor = queries["index"]  # type: ignore
        if do_inner_product_search:
            q_embeddings = torch.nn.functional.normalize(q_embeddings)
        n_devices = torch.cuda.device_count()
        self._q_embeddings = q_embeddings
        self._q_ids = q_ids
        self._n_devices = n_devices
        self._q_embeddings_copy = [
            q_embeddings.to(f"cuda:{i}") for i in range(n_devices)
        ]
        self._q_ids_copy = [q_ids.to(f"cuda:{i}") for i in range(n_devices)]

    def build(self, progress: bool = False) -> Dataset:
        # initialize the top k
        n_q, _ = self._q_embeddings.shape
        shape = (n_q, self._k)
        gt_ids = torch.full(shape, -1, dtype=torch.int32).cuda()
        if self._do_inner_product_search:
            gt_scores = torch.zeros(shape, dtype=torch.float32).cuda()
        else:
            gt_scores = torch.full(shape, torch.inf, dtype=torch.float32).cuda()

        with tqdm(
            desc="make_ground_truth",
            total=len(self._dataset),
            disable=(not progress),
        ) as counter:
            batches = iter_tensors(self._dataset)
            batches, batches_copy = tee(batches, 2)
            lengths = imap(batches_copy, self._get_length, 0)
            batches = imap_multi_gpu(batches, self._local_topk)
            batches = accumulate(
                batches, self._reduce_topk, initial=(gt_ids, gt_scores)
            )
            batches = zip(lengths, batches)
            for length, (gt_ids, _) in batches:
                counter.update(length)

        gt_ids = gt_ids.cpu()

        ground_truth = Dataset.from_dict(
            {
                "embedding": self._q_embeddings,
                "gt_ids": gt_ids.numpy(),
            }
        )
        return ground_truth

    def _get_length(self, ids: torch.Tensor, _: torch.Tensor) -> int:
        return len(ids)

    # NOTE: ground truth is computed with the full embedding length
    def _local_topk(
        self, device: torch.device, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # send to GPU asynchronously
        embeddings = embeddings.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)

        # acquire device copy of queries
        q_ids = self._q_ids_copy[device.index]
        q_embeddings = self._q_embeddings_copy[device.index]

        # don't consider the queries themselves as possible ground truth
        not_in_queries = torch.isin(ids, q_ids, invert=True)
        embeddings = embeddings[not_in_queries]
        ids = ids[not_in_queries]

        if self._do_inner_product_search:
            # ensure that the vectors are unit-length
            embeddings = torch.nn.functional.normalize(embeddings)

            # becomes a matmult for multiple data
            scores = q_embeddings @ embeddings.T
        else:
            # prefer direct calc over following the quadratic form with matmult
            scores = torch.cdist(
                q_embeddings, embeddings, compute_mode="donot_use_mm_for_euclid_dist"
            )

        # only yield k from this batch, in the extreme this k replaces all running k
        top_scores, argtop = torch.topk(
            scores, self._k, dim=1, largest=self._do_inner_product_search
        )
        top_ids = ids[argtop.flatten()].reshape(argtop.shape)

        if self._n_devices > 1:
            return top_ids.cpu(), top_scores.cpu()
        else:
            return top_ids, top_scores  # reduce step is on this GPU

    def _reduce_topk(
        self,
        gt: tuple[torch.Tensor, torch.Tensor],
        batch_top: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_ids, gt_scores = gt
        batch_ids, batch_scores = batch_top

        batch_ids = batch_ids.cuda(non_blocking=True)
        batch_scores = batch_scores.cuda(non_blocking=True)

        # update the top k for each query
        gt_scores = torch.hstack((gt_scores, batch_scores))
        gt_ids = torch.hstack((gt_ids, batch_ids))
        gt_scores, argtop = torch.topk(
            gt_scores, self._k, dim=1, largest=self._do_inner_product_search
        )
        gt_ids = torch.gather(gt_ids, 1, argtop)

        return gt_ids, gt_scores


class GroundTruthProvisioner(Provisioner[Dataset]):
    def __init__(self, **kwargs: Unpack[GroundTruthKwargs]) -> None:
        super().__init__(**kwargs)

    @classmethod
    def with_tune_args(
        cls, dataset: Dataset, queries: Dataset, args: IndexTuneArgs
    ) -> Self:
        # for unit vectors, the L2 minimizing is also the inner-product maximizing
        return cls(
            dataset=dataset,
            queries=queries,
            do_inner_product_search=args.normalize,
            k=args.k,
        )

    def provision(self, progress: bool = False) -> Dataset:
        cache_path = self._compute_cache_path()
        if cache_path.exists():
            return Dataset.load_from_disk(cache_path)
        builder = GroundTruthBuilder(**self._kwargs)
        ground_truth = builder.build(progress=progress)
        ground_truth.save_to_disk(cache_path)
        return ground_truth

    def _compute_cache_filename(self) -> str:
        return f"gt_{self._compute_cache_hash()}"


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


def to_gpu(index: faiss.Index, device: int = 0) -> faiss.Index:
    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True  # float16 is necessary for codes sized 56 bits and over
    env = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(env, device, index, opts)


def to_cpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_gpu_to_cpu(index)


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
            with tqdm(desc="make_index", total=n, disable=(not progress)) as c:
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
                    on_gpus = iunsqueeze(self._on_gpus)
                    for on_cpu in imap(on_gpus, self._transfer_and_reset, -1):
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
        on_gpu.add_with_ids(embeddings.numpy(), ids.numpy())  # type: ignore
        return len(ids)  # yield the number of embeddings added

    def _transfer_and_reset(self, on_gpu: faiss.Index) -> faiss.Index:
        on_cpu = to_cpu(on_gpu)
        on_gpu.reset()
        return on_cpu


class MakeIndexProvisioner(Provisioner[MakeIndexOutput]):
    def __init__(self, **kwargs: Unpack[MakeIndexKwargs]) -> None:
        super().__init__(**kwargs)

    @classmethod
    def with_args(
        cls,
        dataset: Dataset,
        holdouts: torch.Tensor | None,
        args: IndexFillArgs | IndexTuneArgs,
    ) -> Self:
        d = args.dimensions
        if d is None:
            d = len(dataset[0]["embedding"])  # TODO: make this a function?
        return cls(
            empty_index_path=args.empty_index_path,
            dataset=dataset,
            holdouts=holdouts,
            d=d,
            normalize=args.normalize,
        )

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


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, args: IndexTuneArgs
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth["embedding"]  # type: ignore
        gt_ids: npt.NDArray[np.int32] = ground_truth["gt_ids"]  # type: ignore

    if args.dimensions is not None:
        q = q[:, : args.dimensions]
    if args.normalize:
        q = q / np.linalg.norm(q, ord=2, axis=1)[:, np.newaxis]
    gt_ids_int64 = gt_ids.astype(np.int64)  # faiss expects int64

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    if args.one_recall_at_one:
        criterion = faiss.OneRecallAtRCriterion(len(ground_truth), 1)
    else:
        criterion = faiss.IntersectionCriterion(len(ground_truth), args.k)
    criterion.set_groundtruth(None, gt_ids_int64)  # type: ignore (monkey-patched)

    p_space = faiss.ParameterSpace()
    p_space.verbose = args.progress
    p_space.initialize(filled_index)
    results: faiss.OperatingPoints = p_space.explore(  # type: ignore (monkey-patched)
        filled_index, q, criterion
    )

    pareto_vector: faiss.OperatingPointVector = results.optimal_pts
    optimal_params: list[IndexParameters] = []
    for i in range(pareto_vector.size()):
        point: faiss.OperatingPoint = pareto_vector.at(i)
        params = IndexParameters(  # converts from ms to seconds
            recall=point.perf, exec_time=(0.001 * point.t), param_string=point.key
        )
        optimal_params.append(params)

    return optimal_params


def save_ids(path: Path, dataset: Dataset):
    # only the id column is needed to run the index
    dataset.select_columns("id").to_parquet(path, BATCH_SIZE, compression="lz4")


def save_params(
    path: Path,
    dimensions: int | None,
    normalize: bool,
    optimal_params: list[IndexParameters] | None,
):
    params = Params(
        dimensions=dimensions, normalize=normalize, optimal_params=optimal_params
    )
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


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


def ensure_tuned(dataset: Dataset, args: IndexTuneArgs) -> None:
    # the queries is to be held out from the making of a provisional index
    queries = dataset.shuffle(seed=42).skip(len(dataset) - args.queries)

    provisioner = GroundTruthProvisioner.with_tune_args(dataset, queries, args)
    ground_truth = provisioner.provision(progress=args.progress)

    with queries.formatted_as("torch"):
        q_ids: torch.Tensor = queries["index"]  # type: ignore

    provisioner = MakeIndexProvisioner.with_args(dataset, q_ids, args)
    merged_index = provisioner.provision(progress=args.progress).open()

    optimal_params = tune_index(merged_index, ground_truth, args)

    with del_on_exc(args.params_path):
        save_params(args.params_path, args.dimensions, args.normalize, optimal_params)


def ensure_filled(dataset: Dataset, args: IndexFillArgs) -> None:
    provisioner = MakeIndexProvisioner.with_args(dataset, None, args)
    output = provisioner.provision(progress=args.progress)

    index_path, ondisk_path = args.index_paths

    with del_on_exc([args.ids_path, index_path, ondisk_path]):
        save_ids(args.ids_path, dataset)
        copy(output.index_path, index_path)
        copy(output.ondisk_path, ondisk_path)


def index_main(args: AllIndexSubcommandArgs):
    if args.subcommand == "clean":
        clean_persistent_cache()
        if args.source:
            # NOTE: if the cache wasn't created, this will create then delete the cache
            dataset = load_dataset(args.source)
            clean_hf_cache(dataset)
        return 0

    if not args.use_cache:
        seal_hf_cache()
        seal_persistent_cache()

    if not args.progress:
        disable_progress_bars()

    # prepare source dataset and destination directory
    dataset = load_dataset(args.source)
    if not args.build_dir.exists():
        args.build_dir.mkdir()

    # TODO: make these functions return int, and rename them to "main" funcs?
    match args.subcommand:
        case "train":
            ensure_trained(dataset, args)
        case "tune":
            ensure_tuned(dataset, args)
        case "fill":
            ensure_filled(dataset, args)
        case _ as mode:
            assert_never(mode)

    return 0
