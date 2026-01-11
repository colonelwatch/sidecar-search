import json
from pathlib import Path
from typing import TypedDict


class IndexParameters(TypedDict):
    recall: float  # in this case 10-recall@10
    exec_time: float  # seconds (raw faiss measure is in milliseconds)
    param_string: str  # pass directly to faiss index


class Params(TypedDict):
    dimensions: int | None
    normalize: bool
    optimal_params: list[IndexParameters] | None


def save_params(
    path: Path,
    dimensions: int | None,
    normalize: bool,
    optimal_params: list[IndexParameters] | None,
):
    params = Params(
        dimensions=dimensions, normalize=normalize, optimal_params=optimal_params
    )
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
