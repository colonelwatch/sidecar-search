from pathlib import Path
from typing import Generator

import numpy as np
import torch
from datasets import Dataset

BATCH_SIZE = 1024


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in dir.glob("*.parquet")]
    dataset: Dataset = Dataset.from_parquet(paths)  # type: ignore
    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore  (wrong func signature)


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
            yield batch["index"], batch["embedding"]  # type: ignore
