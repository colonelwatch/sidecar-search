import sqlite3
import warnings
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Iterable, LiteralString, Mapping, assert_never

# this module is auto-imported by __init__ to ensure the adapters and converters
# are registered, and a lazy PyTorch import preserves the import time floor
if TYPE_CHECKING:
    import torch

SQLITE3_VECTOR_PREFIX = "sidecarsearch_vector_"

_dtype_code_inverse_mapping: Mapping["torch.dtype", "DTypeCode"] | None = None


class DTypeCode(StrEnum):
    FLOAT = "fp32"
    DOUBLE = "fp64"
    HALF = "fp16"
    BFLOAT16 = "bf16"

    def to_torch(self) -> "torch.dtype":
        import torch

        match self:
            case DTypeCode.FLOAT:
                return torch.float32
            case DTypeCode.DOUBLE:
                return torch.float64
            case DTypeCode.HALF:
                return torch.float16
            case DTypeCode.BFLOAT16:
                return torch.bfloat16
            case _ as unrecognized:
                assert_never(unrecognized)
                raise ValueError(f"unrecognized dtype {unrecognized}")

    @classmethod
    def from_torch(cls, dtype: "torch.dtype") -> "DTypeCode":
        inverse_map = cls._get_inverse_mapping()
        try:
            return inverse_map[dtype]
        except KeyError as e:
            raise ValueError("unrecognized dtype") from e

    def to_sqlite3_decltype(self) -> LiteralString:
        return SQLITE3_VECTOR_PREFIX + self.value

    @classmethod
    def _get_inverse_mapping(cls) -> Mapping["torch.dtype", "DTypeCode"]:
        global _dtype_code_inverse_mapping

        if _dtype_code_inverse_mapping is not None:
            return _dtype_code_inverse_mapping

        inverse_mapping = {value.to_torch(): value for value in cls}

        _dtype_code_inverse_mapping = inverse_mapping
        return inverse_mapping


def create_embeddings_table(conn: sqlite3.Connection, dtype_code: DTypeCode) -> None:
    (page_size,) = conn.execute("PRAGMA page_size").fetchone()
    if page_size < 16384:
        warnings.warn(
            "Current page size is small, and disk usage may be inflated. Use "
            "16384, 32768, or 65536 (if supported) and VACUUM if needed."
        )
    decltype = dtype_code.to_sqlite3_decltype()
    conn.execute(f"CREATE TABLE embeddings(id TEXT PRIMARY KEY, embedding {decltype})")


@dataclass
class _Vector:
    vector: "torch.Tensor"

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(f"expected 1D Tensor, got {self.vector.ndim}D Tensor")

    def to_sqlite3(self) -> memoryview:
        import torch

        return self.vector.view(torch.uint8).numpy().data

    @staticmethod
    def to_torch(blob: bytes, dtype_code: DTypeCode | None = None) -> "torch.Tensor":
        import torch

        # PyTorch does not support read-only buffers, so copy to a bytearray
        dtype = dtype_code.to_torch() if dtype_code else None
        return torch.asarray(bytearray(blob), dtype=dtype)


def insert_embeddings(
    pairs: Iterable[tuple[str, "torch.Tensor"]], conn: sqlite3.Connection
) -> None:
    conn.executemany(
        "INSERT INTO embeddings VALUES(?, ?) "
        "ON CONFLICT(id) DO UPDATE SET embedding=excluded.embedding",
        ((id_, _Vector(embedding)) for id_, embedding in pairs),
    )


# register converters
for dtype_code in DTypeCode:
    sqlite3.register_converter(
        dtype_code.to_sqlite3_decltype(),
        partial(_Vector.to_torch, dtype_code=dtype_code),
    )

# register adapter
sqlite3.register_adapter(_Vector, _Vector.to_sqlite3)
