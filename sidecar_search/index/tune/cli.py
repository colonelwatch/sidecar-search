import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import torch
from datasets import Dataset

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexSharedArgsMixin
from ..make import MakeIndexProvisioner
from ..parameters import Params, save_params
from ..utils.datasets_utils import resolve_dimensions
from .ground_truth import GroundTruthProvisioner, ground_truth_to_faiss
from .tune import serialize_operating_points, tune_index


@dataclass
class IndexTuneArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["tune"]]
):
    source: Path
    intersection: int | None  # 1R@1 else kR@k
    queries: int

    # not args
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

        self.k = self.intersection if self.intersection is not None else 1

        with open(self.untuned_params_path) as f:
            params: Params = json.load(f)
        self.dimensions = params["dimensions"]
        self.normalize = params["normalize"]


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
        q_ids = cast(torch.Tensor, queries._getitem("index"))

    dimensions = resolve_dimensions(dataset, args.dimensions)
    provisioner = MakeIndexProvisioner(
        empty_index_path=args.empty_index_path,
        dataset=dataset,
        holdouts=q_ids,
        d=dimensions,
        normalize=args.normalize,
    )
    merged_index = provisioner.provision(progress=args.progress).open()

    gt_queries, gt_ids = ground_truth_to_faiss(ground_truth, dimensions, args.normalize)
    results = tune_index(
        merged_index, gt_queries, gt_ids, args.intersection, progress=args.progress
    )
    optimal_params = serialize_operating_points(results)

    with del_on_exc(args.params_path):
        save_params(args.params_path, args.dimensions, args.normalize, optimal_params)
