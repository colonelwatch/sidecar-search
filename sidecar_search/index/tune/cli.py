import json
from typing import Self, cast, override

import torch
from pydantic import AliasChoices, BaseModel, DirectoryPath, Field, model_validator
from pydantic_settings import CliPositionalArg

from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexMixin
from ..make import MakeIndexProvisioner
from ..parameters import Params, save_params
from ..utils.datasets_utils import load_dataset, resolve_dimensions
from .ground_truth import GroundTruthProvisioner, ground_truth_to_faiss
from .tune import serialize_operating_points, tune_index


class IndexTune(IndexMixin, BaseModel):
    source: CliPositionalArg[DirectoryPath]
    intersection: int | None = Field(  # 1R1 else kR@k
        None, validation_alias=AliasChoices("k", "intersection")
    )
    queries: int = Field(8192, validation_alias=AliasChoices("q", "queries"))

    @model_validator(mode="after")
    def check_inputs_exist(self) -> Self:
        if not self.empty_index_path.exists():
            raise ValueError(f'empty index "{self.empty_index_path}" does not exist')
        if not self.untuned_params_path.exists():
            raise ValueError(
                f'untuned params "{self.untuned_params_path}" does not exist'
            )
        return self

    @property
    def k(self) -> int:
        return 1 if self.intersection is None else self.intersection

    @override
    def cli_cmd(self) -> None:
        super().cli_cmd()

        with open(self.untuned_params_path) as f:
            params: Params = json.load(f)
        dimensions: int | None = params["dimensions"]
        normalize: bool = params["normalize"]

        dataset = load_dataset(self.source)

        # the queries is to be held out from the making of a provisional index
        queries = dataset.shuffle(seed=42).skip(len(dataset) - self.queries)

        # NOTE: for normalized vectors, L2-minimizing == IP-maximizing
        provisioner = GroundTruthProvisioner(
            dataset=dataset,
            queries=queries,
            do_inner_product_search=normalize,
            k=self.k,
        )
        ground_truth = provisioner.provision(progress=self.progress)

        with queries.formatted_as("torch"):
            q_ids = cast(torch.Tensor, queries._getitem("index"))

        dimensions = resolve_dimensions(dataset, dimensions)
        provisioner = MakeIndexProvisioner(
            empty_index_path=self.empty_index_path,
            dataset=dataset,
            holdouts=q_ids,
            d=dimensions,
            normalize=normalize,
        )
        merged_index = provisioner.provision(progress=self.progress).open()

        gt_queries, gt_ids = ground_truth_to_faiss(ground_truth, dimensions, normalize)
        results = tune_index(
            merged_index, gt_queries, gt_ids, self.intersection, progress=self.progress
        )
        optimal_params = serialize_operating_points(results)

        with del_on_exc(self.params_path):
            save_params(self.params_path, dimensions, normalize, optimal_params)
