from pydantic import BaseModel, DirectoryPath, Field

from sidecar_search.utils.cache_utils import clean_hf_cache, clean_persistent_cache

from ..utils.datasets_utils import load_dataset


class IndexClean(BaseModel):
    """Clean `sidecar-search index` caches, including HuggingFace caches."""

    source: DirectoryPath | None = Field(
        None, description="delete HuggingFace caches associated with this dataset"
    )

    def cli_cmd(self) -> None:
        clean_persistent_cache()
        if self.source:
            # NOTE: if cache wasn't created, this will create then delete it
            dataset = load_dataset(self.source)
            clean_hf_cache(dataset)
