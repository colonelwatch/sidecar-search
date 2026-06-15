from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from sidecar_search.utils.cli_utils import extract_short_description

from .clean.cli import IndexClean
from .fill.cli import IndexFill
from .train.cli import IndexTrain
from .tune.cli import IndexTune


class Index(BaseModel):
    """Tools for building, training, and tuning indexes from embeddings."""

    clean: CliSubCommand[IndexClean] = Field(
        description=extract_short_description(IndexClean)
    )
    train: CliSubCommand[IndexTrain] = Field(
        description=extract_short_description(IndexTrain)
    )
    tune: CliSubCommand[IndexTune] = Field(
        description=extract_short_description(IndexTune)
    )
    fill: CliSubCommand[IndexFill] = Field(
        description=extract_short_description(IndexFill)
    )

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)
