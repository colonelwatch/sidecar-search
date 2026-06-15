import json
from pathlib import Path
from shutil import copy
from typing import Self, override

from datasets import Dataset
from pydantic import BaseModel, DirectoryPath, Field, model_validator
from pydantic_settings import CliPositionalArg

from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexMixin
from ..make import MakeIndexProvisioner
from ..parameters import Params
from ..utils.datasets_utils import BATCH_SIZE, load_dataset, resolve_dimensions


class IndexFill(IndexMixin, BaseModel):
    """Build a filled index from a trained base and a dataset.

    Reads the trained base from the build directory and writes the filled index
    to the build directory.
    """

    source: CliPositionalArg[DirectoryPath] = Field(
        description="HuggingFace dataset to read from"
    )

    @model_validator(mode="after")
    def check_inputs_exist(self) -> Self:
        if not self.empty_index_path.exists():
            raise ValueError(f'empty index "{self.empty_index_path}" does not exist')
        if not self.untuned_params_path.exists():
            raise ValueError(
                f'untuned params "{self.untuned_params_path}" does not exist'
            )
        return self

    @override
    def cli_cmd(self) -> None:
        super().cli_cmd()

        with open(self.untuned_params_path) as f:
            params: Params = json.load(f)
        dimensions = params["dimensions"]
        normalize = params["normalize"]

        dataset = load_dataset(self.source)

        dimensions = resolve_dimensions(dataset, dimensions)
        provisioner = MakeIndexProvisioner(
            empty_index_path=self.empty_index_path,
            dataset=dataset,
            holdouts=None,
            d=dimensions,
            normalize=normalize,
        )
        output = provisioner.provision(progress=self.progress)

        index_path, ondisk_path = self.index_paths

        with del_on_exc([self.ids_path, index_path, ondisk_path]):
            _save_ids(self.ids_path, dataset)
            copy(output.index_path, index_path)
            copy(output.ondisk_path, ondisk_path)


def _save_ids(path: Path, dataset: Dataset):
    # only the id column is needed to run the index
    dataset.select_columns("id").to_parquet(path, BATCH_SIZE, compression="lz4")
