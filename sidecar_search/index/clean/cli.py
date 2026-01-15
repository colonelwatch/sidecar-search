from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.cache_utils import clean_hf_cache, clean_persistent_cache

from ..args import IndexSharedArgsMixin
from ..utils.datasets_utils import load_dataset


@dataclass
class IndexCleanArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["clean"]]
):
    source: Path | None

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-s", "--source", default=None, type=Path)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source is not None and not self.source.exists():
            raise ValueError(f'source at "{self.source}" does not exist')


def index_clean_main(args: IndexCleanArgs) -> int:
    clean_persistent_cache()
    if args.source:
        # NOTE: if the cache wasn't created, this will create then delete the cache
        dataset = load_dataset(args.source)
        clean_hf_cache(dataset)
    return 0
