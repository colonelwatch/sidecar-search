import sqlite3
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import batched, chain
from pathlib import Path
from typing import Generator, Iterable, Self

import torch
from tqdm import tqdm

from sidecar_search.utils.gpu_utils import imap, iunsqueeze
from sidecar_search.utils.table_utils import insert_embeddings, to_sql_binary

from .encode import DocumentEmbeddingBatch, DocumentIdBatch

# TODO: if we're gonna do global registration, shore up to_sql_binary
sqlite3.register_adapter(torch.Tensor, to_sql_binary)


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
        self._counter: tqdm | None = None

    def filter(
        self,
        inputs: Iterable[DocumentIdBatch],
        n_tasks: int = 0,
        progress: bool = False,
    ) -> Generator[DocumentIdBatch, None, None]:
        if progress:
            self._counter = tqdm()

        batches = iunsqueeze(inputs)
        batches = imap(batches, self._filt, n_tasks)
        yield from batched(chain.from_iterable(batches), self._batch_size)

        if self._counter is not None:
            self._counter.close()

    def _filt(self, inputs: DocumentIdBatch) -> Iterable[tuple[str, str]]:
        ids: list[str] = [id_ for id_, _ in inputs]
        existing = set(self._conn.pick_existing(ids))

        if self._counter is not None:
            self._counter.update(len(ids))  # update with the unfiltered count

        return [(id_, document) for id_, document in inputs if id_ not in existing]


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
