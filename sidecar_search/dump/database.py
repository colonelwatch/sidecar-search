import sqlite3
from pathlib import Path
from typing import Generator, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from sidecar_search.utils.table_utils import VectorConverter, query_bf16

from .parquet_utils import open_parquet, write_to_parquet


def _to_arrays(
    ids: list[str], embeddings: list[npt.NDArray]
) -> tuple[pa.Array, pa.Array]:
    dim = embeddings[0].shape[0]
    flattened = np.hstack(embeddings)
    embeddings_arr = pa.FixedSizeListArray.from_arrays(flattened, dim)
    ids_arr = pa.array(ids, pa.string())
    return ids_arr, embeddings_arr


def _to_chunks(
    dataset: sqlite3.Cursor, size: int
) -> Generator[tuple[pa.Array, pa.Array], None, None]:
    ids_batch: list[str] = []
    embeddings_batch: list[npt.NDArray] = []
    for id_, embedding in dataset:
        ids_batch.append(id_)
        embeddings_batch.append(embedding)

        if len(ids_batch) >= size:
            yield _to_arrays(ids_batch, embeddings_batch)
            ids_batch = []
            embeddings_batch = []

    if ids_batch:
        yield _to_arrays(ids_batch, embeddings_batch)


def dump_database(
    source: Path,
    dest: Path,
    shard_size: int,
    row_group_size: int,
    enforce: Literal["bf16", "fp16"] | None = None,
):
    if not (source.suffix == ".sqlite" and dest.suffix == ""):
        raise ValueError("invalid source and dest types")

    # detect the type used in the database
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        bf16 = query_bf16(conn)

    # VectorConverter does torch.bfloat16 to np.float32
    to_dtype = "fp32" if enforce == "bf16" else enforce
    converter = VectorConverter(bf16, to_dtype)
    sqlite3.register_converter("vector", converter.from_sql_binary)

    # To save RAM, push chunks of row_group_size into shards of shard_size one-by-one
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        # get the dimension by querying the first row and checking its length
        embedding = conn.execute("SELECT embedding FROM embeddings LIMIT 1")
        dim = embedding.fetchone()[0].shape[0]

        # iterate through this massive query in chunks
        cursor = conn.execute("SELECT * FROM embeddings ORDER BY rowid")
        chunks = _to_chunks(cursor, row_group_size)

        id_ = 0  # shard id
        counter = 0  # the number of rows the current shard will have
        shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)
        for ids_chunk, embd_chunk in chunks:
            # start by assuming this shard will get the whole chunk
            counter += len(ids_chunk)

            # open new shard(s) and write so that the remainder fits in one shard
            while counter >= shard_size:
                excess = counter - shard_size

                cutoff = len(ids_chunk) - excess  # != shard_size perhaps only at first
                write_to_parquet(ids_chunk[:cutoff], embd_chunk[:cutoff], shard)
                ids_chunk = ids_chunk[cutoff:]
                embd_chunk = embd_chunk[cutoff:]

                id_ += 1
                counter = excess
                shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)

            if counter:  # if counter didn't happen to be a multiple of shard_size
                write_to_parquet(ids_chunk, embd_chunk, shard)
