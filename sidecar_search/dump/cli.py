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
    source: CliPositionalArg[DirectoryPath | FilePath]
    dest: CliPositionalArg[NewPath]
    batch_size: int = Field(1024, validation_alias=AliasChoices("b", "batch-size"))
    shard_size: int = Field(  # under 4GB
        4194304, validation_alias=AliasChoices("s", "shard-size")
    )
    row_group_size: int = 262144
    enforce_dtype: bool = True

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
