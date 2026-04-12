from pathlib import Path
from typing import Generator, cast

import numpy as np
import torch
from datasets import Dataset, List
from datasets.utils.typing import PathLike as HfPathLike  # NOTE: undocumented!

BATCH_SIZE = 1024


def load_dataset(dir: Path) -> Dataset:
    paths = [str(path) for path in sorted(dir.glob("*.parquet"))]
    if not paths:  # schema inference is impossible without at least one file
        raise ValueError("must pass folder with at least one parquet file")

    # NOTE: `.from_parquet` has list[HfPathLike] in its signature, so the intention is
    #       clearly covariance (mixed list makes little sense), but `list` is invariant
    paths = cast(list[HfPathLike], paths)
    dataset = Dataset.from_parquet(paths)
    assert isinstance(dataset, Dataset), (
        "datasets violated documentation about return type"
    )

    ids = np.arange(len(dataset), dtype=np.int32)  # add unique integer IDs for later
    return dataset.add_column("index", ids)  # type: ignore # datasets func sig is wrong


# TODO: resolve this sooner than later
def resolve_dimensions(dataset: Dataset, dimensions: int | None) -> int:
    if dimensions is not None:
        return dimensions

    try:
        feature = dataset.features["embedding"]
    except KeyError as e:
        raise TypeError('dataset features does not contain "embedding" field') from e
    if not isinstance(feature, List):
        raise TypeError('"embedding" field is not list type')

    dimensions = feature.length
    if dimensions < 0:
        raise TypeError('"embedding" field does not have fixed length')

    return dimensions


def iter_tensors(
    dataset: Dataset, batch_size: int | None = None
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    if batch_size is None:
        batch_size = BATCH_SIZE
    cols = dataset.select_columns(["index", "embedding"])
    cols.set_format("torch")
    for batch in cols.iter(batch_size):
        batch = cast(dict[str, torch.Tensor], batch)
        yield batch["index"], batch["embedding"]
