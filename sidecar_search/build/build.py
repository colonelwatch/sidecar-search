from pathlib import Path
from typing import Iterable

from sentence_transformers import SentenceTransformer

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import DocumentIdBatch, encode_pipelined


def build_batched(
    inputs: Iterable[DocumentIdBatch],
    model: SentenceTransformer,
    db_path: Path,
    filter_tasks: int,
    encode_tasks: int,
    encode_batch_size: int,
    progress: bool = False,
) -> None:
    with SharedConnection(db_path) as conn:
        # TODO: refactor rebatching out of ParallelFilter
        parallel_filter = ParallelFilter(conn, encode_batch_size)

        batches = parallel_filter.filter(
            inputs, n_tasks=filter_tasks, progress=progress
        )
        batches = encode_pipelined(batches, model, n_tasks=encode_tasks)

        insert_as_completed(batches, conn, n_tasks=encode_tasks)
