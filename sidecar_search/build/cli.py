import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import batched, chain
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import iqueue

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import PipelinedEncoder, get_model


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


def build_main(args: BuildArgs) -> int:
    with SharedConnection(args.data_path) as conn:
        rows = (json.loads(line) for line in sys.stdin)
        rows = ((row["id"], row["document"]) for row in rows)
        batches = batched(rows, args.filter_batch_size)
        batches = iqueue(batches)

        batches = ParallelFilter(conn).filter(
            batches, n_tasks=args.filter_tasks, progress=args.progress
        )
        batches = _rebatch(batches, args.batch_size)
        batches = iqueue(batches)

        batches = PipelinedEncoder(
            lambda: get_model(MODEL, BF16, TRUST_REMOTE_CODE),
            tasks_per_gpu=args.tasks,
        ).encode(batches)
        batches = iqueue(batches)

        insert_tasks = torch.cuda.device_count() * args.tasks
        insert_as_completed(batches, conn, n_tasks=insert_tasks)

    return 0


def _rebatch[T](
    batches: Iterator[Sequence[T]], batch_size: int
) -> Iterator[Sequence[T]]:
    return batched(chain.from_iterable(batches), batch_size)
