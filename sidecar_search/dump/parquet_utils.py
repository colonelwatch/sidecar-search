from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def open_parquet(path: str | Path, dim: int, bf16: bool) -> pq.ParquetWriter:
    schema: dict[str, pa.DataType] = {"id": pa.string()}
    if bf16:
        # the conversion from bfloat16 to float32 leaves 16 bits of mantissa which are
        # completely zero. Exploit this with byte-stream split and lz4 compression
        schema["embedding"] = pa.list_(pa.float32(), dim)
        writer = pq.ParquetWriter(
            str(path),
            pa.schema(schema),
            compression="lz4",
            use_byte_stream_split=["embedding"],  # type: ignore (documented option)
        )
    else:
        # otherwise, compressing float embeddings isn't worth it
        schema["embedding"] = pa.list_(pa.float16(), dim)
        writer = pq.ParquetWriter(str(path), pa.schema(schema), compression="none")
    return writer


def write_to_parquet(
    ids_chunk: pa.Array, embd_chunk: pa.Array, writer: pq.ParquetWriter
) -> None:
    schema: pa.Schema = writer.schema  # type: ignore # pyarrow-stubs is wrong here
    batch = pa.table([ids_chunk, embd_chunk], schema=schema)
    writer.write_table(batch, row_group_size=len(ids_chunk))
