import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import BinaryIO, Iterable, Literal, cast

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import imap, iunsqueeze

from .build import build_batched
from .encode import DocumentIdBatch, get_model


@dataclass
class BuildArgs(SharedArgsMixin, CommandArgsBase[Literal["build"]]):
    data_path: Path
    tasks: int
    batch_size: int
    filter_tasks: int
    filter_batch_size: int

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)

        parser.description = "Embeds titles and abstracts."

        parser.add_argument("data_path", type=Path)
        parser.add_argument("-t", "--tasks", default=2, type=int)
        parser.add_argument("-b", "--batch-size", default=256, type=int)
        parser.add_argument("--filter-tasks", default=5, type=int)
        parser.add_argument("--filter-batch-size", default=1024, type=int)


def _process_lines_batch(batch: Iterable[bytes]) -> DocumentIdBatch:
    ids: list[str] = []
    documents: list[str] = []
    for line in batch:
        row = json.loads(line)
        ids.append(row["id"])
        documents.append(row["document"])
    return ids, documents


def iter_documents(batch_size: int) -> Iterable[DocumentIdBatch]:
    stdin_cast = cast(BinaryIO, sys.stdin.buffer)
    inputs = iunsqueeze(batched(stdin_cast, batch_size))
    return imap(inputs, _process_lines_batch, 1, prefetch_factor=3)


def build_main(args: BuildArgs) -> int:
    batches = iter_documents(args.filter_batch_size)  # TODO: confusing naming
    build_batched(
        batches,
        lambda: get_model(MODEL, BF16, TRUST_REMOTE_CODE),
        args.data_path,
        args.filter_tasks,
        args.tasks,
        args.batch_size,
        progress=args.progress,
    )
    return 0
