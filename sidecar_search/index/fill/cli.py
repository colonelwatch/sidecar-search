import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy
from typing import Literal

from datasets import Dataset

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexSharedArgsMixin
from ..make import MakeIndexProvisioner
from ..parameters import Params
from ..utils.datasets_utils import BATCH_SIZE, resolve_dimensions


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


def save_ids(path: Path, dataset: Dataset):
    # only the id column is needed to run the index
    dataset.select_columns("id").to_parquet(path, BATCH_SIZE, compression="lz4")


def ensure_filled(dataset: Dataset, args: IndexFillArgs) -> None:
    dimensions = resolve_dimensions(dataset, args.dimensions)
    provisioner = MakeIndexProvisioner(
        empty_index_path=args.empty_index_path,
        dataset=dataset,
        holdouts=None,
        d=dimensions,
        normalize=args.normalize,
    )
    output = provisioner.provision(progress=args.progress)

    index_path, ondisk_path = args.index_paths

    with del_on_exc([args.ids_path, index_path, ondisk_path]):
        save_ids(args.ids_path, dataset)
        copy(output.index_path, index_path)
        copy(output.ondisk_path, ondisk_path)
