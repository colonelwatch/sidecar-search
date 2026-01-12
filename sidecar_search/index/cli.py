from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
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

from .args import IndexSharedArgsMixin
from .fill.cli import IndexFillArgs, ensure_filled
from .train.cli import IndexTrainArgs, ensure_trained
from .tune.cli import IndexTuneArgs, ensure_tuned


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
