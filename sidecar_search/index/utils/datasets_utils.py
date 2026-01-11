from typing import Generator

import torch
from datasets import Dataset

BATCH_SIZE = 1024


def iter_tensors(
    dataset: Dataset,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    with dataset.formatted_as("torch", columns=["index", "embedding"]):
        for batch in dataset.iter(BATCH_SIZE):
            yield batch["index"], batch["embedding"]  # type: ignore
