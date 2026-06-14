from pydantic import BaseModel
from pydantic_settings import CliApp, CliSubCommand

from .clean.cli import IndexClean
from .fill.cli import IndexFill
from .train.cli import IndexTrain
from .tune.cli import IndexTune


class Index(BaseModel):
    clean: CliSubCommand[IndexClean]
    train: CliSubCommand[IndexTrain]
    tune: CliSubCommand[IndexTune]
    fill: CliSubCommand[IndexFill]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)
