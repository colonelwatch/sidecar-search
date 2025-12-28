from argparse import ArgumentParser
from dataclasses import dataclass

from .args_base import ArgsMixinBase


@dataclass
class SharedArgsMixin(ArgsMixinBase):
    progress: bool

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-P", "--progress", action="store_true")
