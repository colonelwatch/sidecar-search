import json
import sqlite3
import sys
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from subprocess import PIPE, Popen
from typing import BinaryIO, Generator, Iterable, Literal, Self, cast

import torch
from filelock import FileLock
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from sidecar_search.args import SharedArgsMixin
from sidecar_search.args_base import CommandArgsBase
from sidecar_search.utils.env_utils import BF16, MODEL, TRUST_REMOTE_CODE
from sidecar_search.utils.gpu_utils import imap, iunsqueeze, iunzip
from sidecar_search.utils.table_utils import insert_embeddings, to_sql_binary

DocumentIdBatch = tuple[list[str], list[str]]
DocumentEmbeddingBatch = tuple[list[str], torch.Tensor]


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


class SharedConnection:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._worker = ThreadPoolExecutor(1)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        fut = self._worker.submit(sqlite3.connect, self._path, autocommit=False)
        self._conn = fut.result()

    def close(self) -> None:
        if self._conn is None:
            raise RuntimeError("closed with a not-open connection")
        fut = self._worker.submit(self._conn.close)
        fut.result()
        self._worker.shutdown()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def pick_existing(self, ids: list[str]) -> list[str]:
        def _pick_existing():
            conn = self._ensure_conn()
            placeholders = ", ".join("?" * len(ids))
            return conn.execute(
                f"SELECT id from embeddings WHERE id IN ({placeholders})", ids
            ).fetchall()

        fut = self._worker.submit(_pick_existing)
        res: list[tuple[str]] = fut.result()

        return [id_ for (id_,) in res]

    def insert_async(
        self, oa_ids: Iterable[str], embeddings: Iterable[torch.Tensor]
    ) -> Future[None]:
        def _insert() -> None:
            conn = self._ensure_conn()
            insert_embeddings(oa_ids, embeddings, conn)
            conn.commit()

        return self._worker.submit(_insert)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("called with a not-open connection")
        return self._conn


class ParallelFilter:
    def __init__(self, conn: SharedConnection, batch_size: int) -> None:
        self._conn = conn
        self._batch_size = batch_size
        self._filtereds: deque[tuple[str, str]] = deque()
        self._counter: tqdm | None = None

    def filter(
        self,
        batches: Iterable[DocumentIdBatch],
        n_tasks: int = 0,
        progress: bool = False,
    ) -> Generator[DocumentIdBatch, None, None]:
        if progress:
            self._counter = tqdm()

        for filtered in imap(batches, self._filt, n_tasks):
            self._filtereds.extend(filtered.items())
            yield from self._roll(False)
        yield from self._roll(True)

        if self._counter is not None:
            self._counter.close()

    def _filt(self, ids: list[str], documents: list[str]):
        batch = {id_: document for id_, document in zip(ids, documents)}
        for id_ in self._conn.pick_existing(ids):
            del batch[id_]

        if self._counter is not None:
            self._counter.update(len(ids))  # update with the unfiltered count

        return batch

    def _roll(self, drain: bool):
        while len(self._filtereds) >= self._batch_size or (drain and self._filtereds):
            ids_out: list[str] = []
            documents_out: list[str] = []
            for _ in range(self._batch_size):
                try:
                    id_, document = self._filtereds.popleft()
                except IndexError:
                    break
                ids_out.append(id_)
                documents_out.append(document)

            yield ids_out, documents_out


# built from SentenceTransformer.encode but with non-blocking CPU-to-GPU transfers
def encode_faster(
    model: SentenceTransformer,
    sentences: list[str],
) -> torch.Tensor:
    model.eval()

    # Tokenize (which yields a dict) then do a non-blocking transfer
    features = {
        k: v.to(model.device, non_blocking=True)
        for k, v in model.tokenize(sentences).items()
    }

    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]

    return embeddings.cpu()


def encode_pipelined(
    batches: Iterable[DocumentIdBatch],
    model: SentenceTransformer,
    n_tasks: int,
) -> Generator[DocumentEmbeddingBatch, None, None]:
    ids_batches, documents_batches = iunzip(batches, 2)
    documents_batches = iunsqueeze(documents_batches)
    embeddings_batches = imap(
        documents_batches, lambda x: encode_faster(model, x), n_tasks
    )
    batches_out = zip(ids_batches, embeddings_batches)
    for ids_batch, embeddings_batch in batches_out:
        yield ids_batch, embeddings_batch


def insert_as_completed(
    batches: Iterable[DocumentEmbeddingBatch], conn: SharedConnection, n_tasks: int
) -> None:
    pending: deque[Future[None]] = deque()

    for ids_batch, embeddings_batch in batches:
        while (pending and pending[0].done()) or len(pending) > n_tasks:
            pending.popleft().result()

        fut = conn.insert_async(ids_batch, embeddings_batch)
        pending.append(fut)

    for fut in pending:
        fut.result()


def build_main(args: BuildArgs) -> int:
    # Get model with file lock to ensure next process will see this one
    with FileLock("/tmp/abstracts-search-gpu.lock"):
        model = get_model(MODEL, BF16, TRUST_REMOTE_CODE)

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        print("error: model doesn't have exact embedding dim", file=sys.stderr)
        exit(1)

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with SharedConnection(args.data_path) as conn:
        parallel_filter = ParallelFilter(conn, args.batch_size)

        batches = iter_documents(args.filter_batch_size)  # TODO: confusing naming
        batches = parallel_filter.filter(
            batches=batches, n_tasks=args.filter_tasks, progress=args.progress
        )
        batches = encode_pipelined(batches, model, args.tasks)

        insert_as_completed(batches, conn, args.tasks)

    return 0
