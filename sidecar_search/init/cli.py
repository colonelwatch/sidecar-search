import sqlite3

from pydantic import BaseModel, Field, NewPath
from pydantic_settings import CliPositionalArg

from sidecar_search.args import CommonMixin
from sidecar_search.utils.env_utils import BF16
from sidecar_search.utils.table_utils import DTypeCode, create_embeddings_table


class Init(CommonMixin, BaseModel):
    """Initialize a SQLite3 database to be used with `sidecar-search build`."""

    target: CliPositionalArg[NewPath] = Field(
        description="creation location for the SQLite3 database file"
    )

    def cli_cmd(self) -> None:
        with sqlite3.connect(self.target) as conn:
            conn.execute("PRAGMA page_size = 32768")
            create_embeddings_table(
                conn, DTypeCode.BFLOAT16 if BF16 else DTypeCode.HALF
            )
