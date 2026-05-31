from itertools import batched, chain
from pathlib import Path
from typing import Callable, Iterator

import torch
from sentence_transformers import SentenceTransformer

from sidecar_search.utils.gpu_utils import iqueue

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import DocumentIdBatch, PipelinedEncoder


def build_batched(
    inputs: Iterator[DocumentIdBatch],
    model_factory: Callable[[], SentenceTransformer],
    db_path: Path,
    filter_tasks: int,
    encode_tasks: int,
    encode_batch_size: int,
    progress: bool = False,
) -> None:
    with SharedConnection(db_path) as conn:
        parallel_filter = ParallelFilter(conn)
        pipelined_encoder = PipelinedEncoder(model_factory, tasks_per_gpu=encode_tasks)

        inputs = iqueue(inputs)
        batches = parallel_filter.filter(
            inputs, n_tasks=filter_tasks, progress=progress
        )
        batches = batched(chain.from_iterable(batches), encode_batch_size)
        batches = iqueue(batches)
        batches = pipelined_encoder.encode(batches)
        batches = iqueue(batches)

        insert_tasks = torch.cuda.device_count() * encode_tasks
        insert_as_completed(batches, conn, n_tasks=insert_tasks)
