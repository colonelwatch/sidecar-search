import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import faiss  # many monkey-patches, see faiss/python/class_wrappers.py in faiss repo
import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexSharedArgsMixin
from ..make import MakeIndexProvisioner
from ..parameters import IndexParameters, Params, save_params
from ..utils.datasets_utils import resolve_dimensions
from .ground_truth import GroundTruthProvisioner


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

    # NOTE: for normalized vectors, L2-minimizing == IP-maximizing
    provisioner = GroundTruthProvisioner(
        dataset=dataset,
        queries=queries,
        do_inner_product_search=args.normalize,
        k=args.k,
    )
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
