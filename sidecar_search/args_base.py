from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from types import get_original_bases
from typing import (
    ClassVar,
    Iterable,
    Literal,
    LiteralString,
    Mapping,
    cast,
    get_args,
    get_origin,
)


class ConfiguresParser(ABC):
    @classmethod
    @abstractmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        # NOTE: intermediate implementations must call super().configure_parser
        raise RuntimeError(f"mro for {cls.__name__} did not terminate in ArgsBase")


# TODO: check whether ArgsBase is in the MRO at all?
# TODO: check for opt and longopt conflicts
# TODO: document these classes!
# TODO: use friendly messages
# TODO: validate command and subcommand after all? validate the rest too
# NOTE: generic inheritance is not supported until TypeVar resolving is solved
@dataclass
class ArgsBase[T: LiteralString](ConfiguresParser, ABC):
    type_: type["ArgsBase"]
    command: T

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if issubclass(get_origin(cls) or cls, ArgsBase):
            cls._validate_mro()

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        parser.set_defaults(type_=cls)

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> "ArgsBase":
        # NOTE: with correct inheritance, namespace is 1:1 with fields of type_
        type_: type[ArgsBase] = namespace.type_
        return type_(**vars(namespace))

    # TODO: move to a new root class?
    @classmethod
    def _add_commands(
        cls,
        parser: ArgumentParser,
        all_cls_args: Iterable[type["CommandArgsBase | CommandGroupArgsBase"]],
    ) -> None:
        cls_args_mapping: dict[
            LiteralString, type[CommandArgsBase | CommandGroupArgsBase]
        ] = {cls_args.expected_command: cls_args for cls_args in all_cls_args}
        cls._add_subparsers(parser, cls_args_mapping, "command")

    @staticmethod
    def _add_subparsers(
        parser: ArgumentParser,
        cls_args_mapping: Mapping[LiteralString, type["ArgsBase"]],
        field: str,
    ) -> None:
        subparsers = parser.add_subparsers(title=field, dest=field, required=True)
        for command, AnyCommandArgsBase in cls_args_mapping.items():
            subparser = subparsers.add_parser(command)
            AnyCommandArgsBase.configure_parser(subparser)

    @classmethod
    def _validate_mro(cls) -> None:
        past_end = False
        for curr_cls in cls.__mro__:
            curr_cls_generic = get_origin(curr_cls) or curr_cls
            if curr_cls_generic is ArgsBase:
                past_end = True
                continue
            elif curr_cls_generic is ConfiguresParser:
                return
            elif past_end and issubclass(curr_cls_generic, ConfiguresParser):
                raise ValueError(
                    "ArgsBase must be the last ConfiguresParser in the mro"
                )
        assert past_end, "failed to find ArgsBase in mro"

    @classmethod
    def _get_expected_fields(cls, lookup_cls: type) -> tuple[LiteralString, ...]:
        found_cls = None
        for curr_cls in cls.__mro__:
            if curr_cls is object:
                break
            for curr_cls_base in get_original_bases(curr_cls):
                if get_origin(curr_cls_base) is lookup_cls:
                    found_cls = curr_cls_base
                    break
            if found_cls:
                break

        if not found_cls:
            raise RuntimeError(
                f"failed to find {lookup_cls.__name__} in mro of {cls.__name__}"
            )

        fields: list[LiteralString] = []
        for command_type in get_args(found_cls):
            if get_origin(command_type) is not Literal:
                as_str = (
                    command_type.__name__
                    if isinstance(command_type, type)
                    else command_type  # may be a TypeVar
                )
                raise TypeError(f"Expected LiteralString for T, but got {as_str}")
            (command,) = get_args(command_type)
            if not isinstance(command, str):
                raise TypeError(
                    "Expected LiteralString for T, but got Literal with value "
                    f"{command}"
                )
            fields.append(cast(LiteralString, command))

        return tuple(fields)


@dataclass
class CommandArgsBase[T: LiteralString](ArgsBase[T]):
    expected_command: ClassVar[LiteralString]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        (cls.expected_command,) = cls._get_expected_fields(CommandArgsBase)


@dataclass
class CommandGroupArgsBase[T: LiteralString](ArgsBase[T]):
    expected_command: ClassVar[LiteralString]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        (cls.expected_command,) = cls._get_expected_fields(CommandGroupArgsBase)

    @classmethod
    def _add_subcommands(
        cls, parser: ArgumentParser, all_cls_args: Iterable[type["SubcommandArgsBase"]]
    ) -> None:
        cls_args_mapping: dict[LiteralString, type[SubcommandArgsBase]] = {
            cls_args.expected_subcommand: cls_args for cls_args in all_cls_args
        }
        cls._add_subparsers(parser, cls_args_mapping, "subcommand")


@dataclass
class SubcommandArgsBase[T: LiteralString, U: LiteralString](ArgsBase[T]):
    expected_command: ClassVar[LiteralString]
    expected_subcommand: ClassVar[LiteralString]
    subcommand: U

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.expected_command, cls.expected_subcommand = cls._get_expected_fields(
            SubcommandArgsBase
        )


class ArgsMixinBase(ConfiguresParser):
    @classmethod
    @abstractmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
