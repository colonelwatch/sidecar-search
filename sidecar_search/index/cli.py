from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Literal, assert_never, cast, get_args

from datasets import disable_progress_bars

from sidecar_search.args_base import CommandGroupArgsBase
from sidecar_search.utils.cache_utils import seal_hf_cache, seal_persistent_cache

from .args import IndexSharedArgsMixin
from .clean.cli import IndexCleanArgs, index_clean_main
from .fill.cli import IndexFillArgs, ensure_filled
from .train.cli import IndexTrainArgs, ensure_trained
from .tune.cli import IndexTuneArgs, ensure_tuned
from .utils.datasets_utils import load_dataset

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


def index_main(args: AllIndexSubcommandArgs):
    if args.subcommand == "clean":
        return index_clean_main(args)

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
