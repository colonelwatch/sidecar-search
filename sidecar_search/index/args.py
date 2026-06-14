from abc import ABC, abstractmethod
from pathlib import Path

from datasets import disable_progress_bars
from pydantic import AliasChoices, BaseModel, DirectoryPath, Field, NewPath

from sidecar_search.args import CommonMixin
from sidecar_search.utils.cache_utils import seal_hf_cache, seal_persistent_cache


class IndexMixin(CommonMixin, BaseModel, ABC):
    build_dir: NewPath | DirectoryPath = Field(
        Path("."), validation_alias=AliasChoices("B", "build-dir")
    )
    use_cache: bool = False  # for experiments only

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

    @abstractmethod
    def cli_cmd(self) -> None:
        if not self.use_cache:
            seal_hf_cache()
            seal_persistent_cache()
        if not self.progress:
            disable_progress_bars()
        self.build_dir.mkdir(exist_ok=True)
