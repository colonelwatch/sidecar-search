from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from .build.cli import Build
from .dump.cli import Dump
from .index.cli import Index
from .init.cli import Init


class SidecarSearch(BaseSettings):
    init: CliSubCommand[Init]
    build: CliSubCommand[Build]
    index: CliSubCommand[Index]
    dump: CliSubCommand[Dump]

    model_config = SettingsConfigDict(
        frozen=True,
        case_sensitive=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
        cli_implicit_flags="toggle",
        cli_kebab_case=True,
    )

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> int:
    CliApp.run(SidecarSearch)
    return 0
