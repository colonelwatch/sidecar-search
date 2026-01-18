import sqlite3
from pathlib import Path
from typing import Literal, cast

import numpy.typing as npt
import pyarrow as pa
import pyarrow.dataset as ds
import torch

from sidecar_search.utils.table_utils import (
    create_embeddings_table,
    insert_embeddings,
    to_sql_binary,
)


def dump_dataset(
    source: Path,
    dest: Path,
    batch_size: int,
    enforce: Literal["bf16", "fp16"] | None = None,
) -> None:
    if not (source.suffix == "" and dest.suffix == ".sqlite"):
        raise ValueError("invalid source and dest types")

    paths = [str(path) for path in source.glob("*.parquet")]
    dataset: ds.Dataset = ds.dataset(paths)

    # extract the vector dtype and length from the schema
    embeddings_col_type = dataset.schema.field("embedding").type
    dtype = embeddings_col_type.value_type
    length = embeddings_col_type.list_size  # poorly documented!

    if not enforce:
        if dtype == pa.float32():
            bf16 = True  # assume this was converted from bfloat16
        elif dtype == pa.float16():
            bf16 = False
        else:
            raise ValueError(f'invalid embeddings type "{dtype}"')
    else:
        bf16 = True if enforce == "bf16" else False

    sqlite3.register_adapter(torch.Tensor, to_sql_binary)
    with sqlite3.connect(dest) as conn:
        create_embeddings_table(conn, bf16)
        for batch in dataset.to_batches(batch_size=batch_size):
            ids_arr = batch["id"]
            if ids_arr.type != pa.string():
                raise ValueError('"id" batch was not array of StringScalar')
            if ids_arr.null_count != 0:
                raise ValueError('nulls detected in "id" batch')
            ids_list = cast(list[str], ids_arr.to_pylist())

            embeddings_arr = batch["embedding"]
            if not isinstance(embeddings_arr, pa.FixedSizeListArray):
                raise ValueError('"embedding" batch was not array of FixedSizeList')
            embeddings_np: npt.NDArray = (  # this makes the conversion zero-copy
                embeddings_arr.flatten().to_numpy().reshape((-1, length))
            )
            embeddings = torch.from_numpy(embeddings_np.copy())  # no read-only memory
            if bf16:
                embeddings = embeddings.bfloat16()

            insert_embeddings(ids_list, embeddings, conn)
