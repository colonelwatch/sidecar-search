import json
import sys
from itertools import batched, chain
from typing import Iterator, Sequence

import torch
from pydantic import AliasChoices, BaseModel, Field, FilePath
from pydantic_settings import CliPositionalArg

from sidecar_search.args import CommonMixin
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import iqueue

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import PipelinedEncoder, get_model


class Build(CommonMixin, BaseModel):
    data_path: CliPositionalArg[FilePath]
    tasks: int = Field(2, validation_alias=AliasChoices("t", "tasks"))
    batch_size: int = Field(256, validation_alias=AliasChoices("b", "batch-size"))
    filter_tasks: int = 5
    filter_batch_size: int = 1024

    def cli_cmd(self) -> None:
        with SharedConnection(self.data_path) as conn:
            rows = (json.loads(line) for line in sys.stdin)
            rows = ((row["id"], row["document"]) for row in rows)
            batches = batched(rows, self.filter_batch_size)
            batches = iqueue(batches)

            batches = ParallelFilter(conn).filter(
                batches, n_tasks=self.filter_tasks, progress=self.progress
            )
            batches = _rebatch(batches, self.batch_size)
            batches = iqueue(batches)

            batches = PipelinedEncoder(
                lambda: get_model(MODEL, BF16, TRUST_REMOTE_CODE),
                tasks_per_gpu=self.tasks,
            ).encode(batches)
            batches = iqueue(batches)

            insert_tasks = torch.cuda.device_count() * self.tasks
            insert_as_completed(batches, conn, n_tasks=insert_tasks)


def _rebatch[T](
    batches: Iterator[Sequence[T]], batch_size: int
) -> Iterator[Sequence[T]]:
    return batched(chain.from_iterable(batches), batch_size)
