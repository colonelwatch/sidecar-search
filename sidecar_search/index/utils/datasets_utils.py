from pathlib import Path
from typing import TYPE_CHECKING, Generator, cast

import numpy as np
import torch
from datasets import Dataset

if TYPE_CHECKING:
    from datasets.utils.typing import PathLike as HfPathLike  # NOTE: undocumented!

BATCH_SIZE = 1024


def load_dataset(dir: Path) -> Dataset:
    paths: list[HfPathLike] = [str(path) for path in dir.glob("*.parquet")]
    dataset = Dataset.from_parquet(paths)
    assert isinstance(dataset, Dataset), (
        "datasets violated documentation about return type"
    )
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore # datasets func sig is wrong


# TODO: resolve this sooner than later
def resolve_dimensions(dataset: Dataset, dimensions: int | None) -> int:
    if dimensions is None:
        dimensions = len(dataset[0]["embedding"])
    return dimensions


def iter_tensors(
    dataset: Dataset,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    with dataset.formatted_as("torch", columns=["index", "embedding"]):
        for batch in dataset.iter(BATCH_SIZE):
            batch = cast(dict[str, torch.Tensor], batch)
            yield batch["index"], batch["embedding"]
