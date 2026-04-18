import sqlite3
from itertools import batched
from pathlib import Path
from typing import Generator, Literal, Sequence

import pyarrow as pa
import torch

from .parquet_utils import open_parquet, write_to_parquet


def _to_arrays(
    tups: Sequence[tuple[str, torch.Tensor]],
) -> tuple[pa.Array, pa.Array]:
    if not tups:
        raise ValueError("cannot convert empty batch")

    _, embedding_0 = tups[0]
    (dim,) = embedding_0.shape
    dtype = embedding_0.dtype
    if dtype == torch.bfloat16:
        dtype = torch.float32  # bfloat16 does not have a Arrow equivalent

    embeddings_arr = torch.empty((len(tups), dim), dtype=dtype)

    ids: list[str] = []
    for i, (id_, embedding) in enumerate(tups):
        ids.append(id_)
        embeddings_arr[i, :] = embedding

    embeddings_arr = torch.ravel(embeddings_arr)
    embeddings_arr = pa.array(embeddings_arr.numpy())
    embeddings_arr = pa.FixedSizeListArray.from_arrays(embeddings_arr, dim)
    ids_arr = pa.array(ids, pa.string())

    return ids_arr, embeddings_arr


def _to_chunks(
    conn: sqlite3.Connection, size: int
) -> Generator[tuple[pa.Array, pa.Array], None, None]:
    cursor = conn.execute("SELECT * FROM embeddings ORDER BY rowid")
    batches = batched(cursor, size)
    for batch in batches:
        yield _to_arrays(batch)


def dump_database(
    source: Path,
    dest: Path,
    shard_size: int,
    row_group_size: int,
    enforce: Literal["bf16", "fp16"] | None = None,
):
    if not (source.suffix == ".sqlite" and dest.suffix == ""):
        raise ValueError("invalid source and dest types")

    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.execute("SELECT embedding FROM embeddings LIMIT 1")
        (embedding,) = cursor.fetchone()

        (dim,) = embedding.shape
        dtype = embedding.dtype

    if not enforce:
        if dtype == torch.bfloat16:
            bf16 = True
        elif dtype == torch.float16:
            bf16 = False
        else:
            raise ValueError(f"unrecognized embedding dtype {dtype}")
    else:
        bf16 = enforce == "bf16"

    # To save RAM, push chunks of row_group_size into shards of shard_size one-by-one
    with sqlite3.connect(source, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        id_ = 0  # shard id
        counter = 0  # the number of rows the current shard will have
        shard = open_parquet(dest / f"data_{id_:03}.parquet", dim, bf16)
        for ids_chunk, embd_chunk in _to_chunks(conn, row_group_size):
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
