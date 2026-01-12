from typing import Generator

import torch
from datasets import Dataset

BATCH_SIZE = 1024


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
