import json
import sys
from collections.abc import Iterator, Sequence
from itertools import batched, chain

import torch
from pydantic import AliasChoices, BaseModel, Field, FilePath
from pydantic_settings import CliPositionalArg

from sidecar_search.args import CommonMixin
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import iqueue

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import PipelinedEncoder, get_model


class Build(CommonMixin, BaseModel):
    """Encode ID-document pairs as ID-embedding pairs and store them.

    Reads ID-document pairs as lines of `{"id": "...", "document": "..."}` JSON
    objects over stdin, encode them, and store them in a SQLite3 database at
    DEST, committing incrementally. Committed ID-embedding pairs are not lost
    upon interruption of the process.

    If an ID is already in the database, no encoding is performed for that
    document, and the database will not be updated for that ID.
    """

    dest: CliPositionalArg[FilePath] = Field(
        description=(
            "the name of the SQLite database, must already exist "
            "(initialized by `sidecar-search init`)"
        )
    )
    tasks: int = Field(
        2,
        validation_alias=AliasChoices("t", "tasks"),
        description="number of concurrent encode tasks per gpu",
    )
    batch_size: int = Field(
        256,
        validation_alias=AliasChoices("b", "batch-size"),
        description=(
            "size of encode batches, trades off memory usage "
            "for reduced encode overhead/bottlenecking"
        ),
    )
    filter_tasks: int = Field(
        5, description="number of DB filter and number of DB insert tasks"
    )
    filter_batch_size: int = Field(
        1024, description="size of filter and insert batches"
    )

    def cli_cmd(self) -> None:
        with SharedConnection(self.dest) as conn:
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
