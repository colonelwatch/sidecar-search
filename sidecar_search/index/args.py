from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from sidecar_search.args import SharedArgsMixin


@dataclass
class IndexSharedArgsMixin(SharedArgsMixin):
    build_dir: Path
    use_cache: bool  # for experiments only

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("-B", "--build-dir", default=Path("."), type=Path)
        parser.add_argument("--use-cache", action="store_true")

    # TODO: rethink __post_init__ inheritance
    def __post_init__(self) -> None:
        if self.build_dir.exists() and not self.build_dir.is_dir():
            raise ValueError(
                f'build dir "{self.build_dir}" exists but is not a directory'
            )

    @property
    def empty_index_path(self) -> Path:
        return self.build_dir / "empty.faiss"

    @property
    def untuned_params_path(self) -> Path:
        return self.build_dir / "untuned.json"

    @property
    def params_path(self) -> Path:
        return self.build_dir / "params.json"

    @property
    def ids_path(self) -> Path:
        return self.build_dir / "ids.parquet"

    @property
    def index_paths(self) -> tuple[Path, Path]:
        return self.build_dir / "index.faiss", self.build_dir / "ondisk.ivfdata"
