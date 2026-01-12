import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy
from typing import Literal, assert_never, cast, get_args

import numpy as np
from datasets import Dataset, disable_progress_bars

from sidecar_search.args_base import CommandGroupArgsBase, SubcommandArgsBase
from sidecar_search.utils.cache_utils import (
    clean_hf_cache,
    clean_persistent_cache,
    seal_hf_cache,
    seal_persistent_cache,
)
from sidecar_search.utils.contextmanager_utils import del_on_exc

from .args import IndexSharedArgsMixin
from .make import MakeIndexProvisioner
from .parameters import Params
from .train.cli import IndexTrainArgs, ensure_trained
from .tune.cli import IndexTuneArgs, ensure_tuned
from .utils.datasets_utils import BATCH_SIZE, resolve_dimensions


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


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore  (wrong func signature)


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
