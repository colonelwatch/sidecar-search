from shutil import rmtree
from typing import Self

from pydantic import (
    AliasChoices,
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    NewPath,
    PrivateAttr,
    model_validator,
)
from pydantic_settings import CliPositionalArg

from sidecar_search.args import CommonMixin
from sidecar_search.utils.env_utils import BF16

from .database import dump_database
from .dataset import dump_dataset


class Dump(CommonMixin, BaseModel):
    """Dump ID-embedding pairs from SOURCE to DEST, converting formats.

    The only supported input/output formats are HuggingFace (HF) datasets to
    SQLite3 databases and SQLite3 databases to HF datasets. HF datasets to HF
    datasets and SQLite3 databases to SQLite3 databases are not supported.
    """

    source: CliPositionalArg[DirectoryPath | FilePath] = Field(
        description="HuggingFace dataset / SQLite3 database to read from"
    )
    dest: CliPositionalArg[NewPath] = Field(
        description=(
            "HuggingFace dataset / SQLite3 database to write to. "
            "Must not be an existing folder/file"
        )
    )
    batch_size: int = Field(
        1024,
        validation_alias=AliasChoices("b", "batch-size"),
        description="read/write batch size",
    )
    shard_size: int = Field(  # under 4GB
        4194304,
        validation_alias=AliasChoices("s", "shard-size"),
        description=(
            "size of dataset shards, in number of vectors. "
            "Only applies to writing datasets"
        ),
    )
    row_group_size: int = Field(
        262144,
        description=(
            "Parquet row group size, in number of vectors. "
            "Only applies to writing datasets"
        ),
    )
    no_enforce_dtype: bool = Field(False, description="disable enforcing data type")

    _to_dataset: bool = PrivateAttr()

    @model_validator(mode="after")
    def check_direction(self) -> Self:
        if self.source.suffix == ".sqlite" and self.dest.suffix == "":
            self._to_dataset = True
        elif self.source.suffix == "" and self.dest.suffix == ".sqlite":
            self._to_dataset = False
        else:
            raise ValueError("invalid source and destination types")
        return self

    @property
    def enforce_dtype(self) -> bool:
        return not self.no_enforce_dtype

    def cli_cmd(self) -> None:
        source = self.source
        dest = self.dest

        if self.enforce_dtype:
            enforce = "bf16" if BF16 else "fp16"
        else:
            enforce = None

        if self._to_dataset:
            dest.mkdir()
            try:
                dump_database(
                    source, dest, self.shard_size, self.row_group_size, enforce
                )
            except (KeyboardInterrupt, Exception):
                rmtree(dest)
                raise
        else:
            try:
                dump_dataset(source, dest, self.batch_size, enforce)
            except (KeyboardInterrupt, Exception):
                dest.unlink()
                raise
