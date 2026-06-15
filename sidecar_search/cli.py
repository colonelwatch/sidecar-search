from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from sidecar_search.utils.cli_utils import extract_short_description

from .build.cli import Build
from .dump.cli import Dump
from .index.cli import Index
from .init.cli import Init


class SidecarSearch(BaseSettings):
    """CLI build tools for sidecar indexes to add semantic search to anything."""

    init: CliSubCommand[Init] = Field(description=extract_short_description(Init))
    build: CliSubCommand[Build] = Field(description=extract_short_description(Build))
    index: CliSubCommand[Index] = Field(description=extract_short_description(Index))
    dump: CliSubCommand[Dump] = Field(description=extract_short_description(Dump))

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
