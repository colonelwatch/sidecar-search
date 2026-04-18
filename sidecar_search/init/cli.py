import sqlite3
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16
from sidecar_search.utils.table_utils import DTypeCode, create_embeddings_table


@dataclass
class InitArgs(SharedArgsMixin, CommandArgsBase[Literal["init"]]):
    target: Path

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("target", type=Path)

    def __post_init__(self) -> None:
        if self.target.exists():
            raise ValueError(f'target "{self.target}" exists')


def init_main(args: InitArgs) -> int:
    with sqlite3.connect(args.target) as conn:
        conn.execute("PRAGMA page_size = 32768")
        create_embeddings_table(conn, DTypeCode.BFLOAT16 if BF16 else DTypeCode.HALF)
    return 0
