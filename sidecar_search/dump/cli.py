from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from sys import stderr
from typing import Literal

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16

from .database import dump_database
from .dataset import dump_dataset


@dataclass
class DumpArgs(SharedArgsMixin, CommandArgsBase[Literal["dump"]]):
    source: Path
    dest: Path
    batch_size: int
    shard_size: int
    row_group_size: int
    enforce_dtype: bool

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("source", type=Path)
        parser.add_argument("dest", type=Path)
        parser.add_argument("-b", "--batch-size", default=1024, type=int)
        parser.add_argument(
            "-s", "--shard-size", default=4194304, type=int
        )  # under 4GB
        parser.add_argument(
            "--row-group-size", default=262144, type=int
        )  # around 128MB
        parser.add_argument(
            "--no-enforce-dtype", action="store_false", dest="enforce_dtype"
        )

    def __post_init__(self) -> None:
        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')
        if self.dest.exists():
            raise ValueError(f'destination path "{self.dest}" exists')


def dump_main(args: DumpArgs) -> int:
    source = args.source
    dest = args.dest

    if args.enforce_dtype:
        enforce = "bf16" if BF16 else "fp16"
    else:
        enforce = None

    if source.suffix == ".sqlite" and dest.suffix == "":
        dest.mkdir()
        try:
            dump_database(source, dest, args.shard_size, args.row_group_size, enforce)
        except (KeyboardInterrupt, Exception):
            rmtree(dest)
            raise
    elif source.suffix == "" and dest.suffix == ".sqlite":
        try:
            dump_dataset(source, dest, args.batch_size, enforce)
        except (KeyboardInterrupt, Exception):
            dest.unlink()
            raise
    else:
        print("error: invalid source and destination types", file=stderr)
        return 1

    return 0
