import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from itertools import accumulate, tee
from pathlib import Path
from typing import Literal, Self, TypedDict, Unpack

import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from tqdm import tqdm

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc
from sidecar_search.utils.gpu_utils import imap, imap_multi_gpu

from ..args import IndexSharedArgsMixin
from ..make import MakeIndexProvisioner
from ..parameters import IndexParameters, Params, save_params
from ..provisioner import Provisioner
from ..utils.datasets_utils import iter_tensors, resolve_dimensions


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
        parser.add_argument("source", type=Path)
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
        # TODO: calling _getitem is a workaround, bypassing the Column class
        #       used since datasets 4.0.0, but this should be rewritten sometime
        with queries.formatted_as("torch", columns=["embedding", "index"]):
            q_embeddings: torch.Tensor = queries._getitem("embedding")  # type: ignore
            q_ids: torch.Tensor = queries._getitem("index")  # type: ignore

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


def tune_index(
    filled_index: faiss.Index, ground_truth: Dataset, args: IndexTuneArgs
) -> list[IndexParameters]:
    with ground_truth.formatted_as("numpy"):
        q: npt.NDArray[np.float32] = ground_truth._getitem("embedding")  # type: ignore
        gt_ids: npt.NDArray[np.int32] = ground_truth._getitem("gt_ids")  # type: ignore

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


def ensure_tuned(dataset: Dataset, args: IndexTuneArgs) -> None:
    # the queries is to be held out from the making of a provisional index
    queries = dataset.shuffle(seed=42).skip(len(dataset) - args.queries)

    provisioner = GroundTruthProvisioner.with_tune_args(dataset, queries, args)
    ground_truth = provisioner.provision(progress=args.progress)

    with queries.formatted_as("torch"):
        q_ids: torch.Tensor = queries._getitem("index")  # type: ignore

    dimensions = resolve_dimensions(dataset, args.dimensions)
    provisioner = MakeIndexProvisioner(
        empty_index_path=args.empty_index_path,
        dataset=dataset,
        holdouts=q_ids,
        d=dimensions,
        normalize=args.normalize,
    )
    merged_index = provisioner.provision(progress=args.progress).open()

    optimal_params = tune_index(merged_index, ground_truth, args)

    with del_on_exc(args.params_path):
        save_params(args.params_path, args.dimensions, args.normalize, optimal_params)
