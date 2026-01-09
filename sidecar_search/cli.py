from argparse import ArgumentParser
from dataclasses import dataclass
from sys import stderr
from typing import assert_never, cast, get_args

from .args import SharedArgsMixin
from .args_base import ArgsBase
from .build.cli import BuildArgs, build_main
from .dump.cli import DumpArgs, dump_main
from .index import AllIndexSubcommandArgs, IndexGroupArgs, index_main

AllCommandArgs = BuildArgs | IndexGroupArgs | DumpArgs
AllArgs = BuildArgs | AllIndexSubcommandArgs | DumpArgs

ALL_COMMAND_ARGS = cast(tuple[type[AllCommandArgs], ...], get_args(AllCommandArgs))


# TODO: make pre-configured parser in args_base?
# TODO: add prog, description, etc
def make_parser() -> ArgumentParser:
    return ArgumentParser(conflict_handler="error")


@dataclass
class CliArgs(SharedArgsMixin, ArgsBase):
    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        cls._add_commands(parser, ALL_COMMAND_ARGS)


def main() -> int:
    parser = make_parser()
    CliArgs.configure_parser(parser)
    args = parser.parse_args()
    try:
        args = cast(AllArgs, CliArgs.from_namespace(args))
    except ValueError as e:  # TODO: make my own exception class?
        print("error:", e.args[0], file=stderr)
        return 1

    match args.command:
        case "build":
            ret = build_main(args)
        case "dump":
            ret = dump_main(args)
        case "index":
            ret = index_main(args)
        case _:
            assert_never(args.command)

    return ret


# TODO: delete this
if __name__ == "__main__":
    exit(main())
