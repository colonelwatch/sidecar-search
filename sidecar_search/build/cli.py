import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from subprocess import PIPE, Popen
from typing import BinaryIO, Iterable, Literal, cast

import torch
from filelock import FileLock
from sentence_transformers import SentenceTransformer

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import imap, iunsqueeze

from .db import ParallelFilter, SharedConnection, insert_as_completed
from .encode import DocumentIdBatch, encode_pipelined


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


def get_model(
    model_name: str, bf16: bool, trust_remote_code: bool
) -> SentenceTransformer:
    # start queries in parallel
    p1 = Popen(
        ["nvidia-smi", "--query-gpu=gpu_bus_id,index", "--format=csv,noheader"],
        stdout=PIPE,
    )
    p2 = Popen(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,name", "--format=csv,noheader"],
        stdout=PIPE,
    )
    assert p1.stdout is not None
    assert p2.stdout is not None

    # get "cuda:X" device indices for each GPU
    bus_id_to_index: dict[str, int] = {}
    with p1:
        for line in p1.stdout:
            gpu_bus_id, index = [v.strip() for v in line.decode().split(",")]
            bus_id_to_index[gpu_bus_id] = int(index)

    # get the processes on each "cuda:X" device
    proc_count = [0] * len(bus_id_to_index)
    with p2:
        for line in p2.stdout:
            gpu_bus_id, proc_name = [v.strip() for v in line.decode().split(",")]
            if "python" in proc_name:
                proc_count[bus_id_to_index[gpu_bus_id]] += 1

    # Find the first device that isn't occupied by python then occupy it with the model
    selected_index = min(range(len(proc_count)), key=(lambda i: proc_count[i]))
    model = SentenceTransformer(
        model_name,
        device=f"cuda:{selected_index}",
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch.bfloat16 if bf16 else torch.float16},
    )

    return model


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
    # Get model with file lock to ensure next process will see this one
    with FileLock("/tmp/abstracts-search-gpu.lock"):
        model = get_model(MODEL, BF16, TRUST_REMOTE_CODE)

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        print("error: model doesn't have exact embedding dim", file=sys.stderr)
        exit(1)

    with SharedConnection(args.data_path) as conn:
        parallel_filter = ParallelFilter(conn, args.batch_size)

        batches = iter_documents(args.filter_batch_size)  # TODO: confusing naming
        batches = parallel_filter.filter(
            inputs=batches, n_tasks=args.filter_tasks, progress=args.progress
        )
        batches = encode_pipelined(batches, model, args.tasks)

        insert_as_completed(batches, conn, args.tasks)

    return 0
