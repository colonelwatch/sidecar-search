import re
from typing import Self, override

import faiss
import numpy as np
from datasets import Dataset
from pydantic import (
    AliasChoices,
    BaseModel,
    DirectoryPath,
    Field,
    PrivateAttr,
    model_validator,
)
from pydantic_settings import CliPositionalArg

from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexMixin
from ..parameters import save_params
from ..utils.datasets_utils import load_dataset, resolve_dimensions
from ..utils.faiss_utils import to_cpu, to_gpu
from .memmap import MemmapProvisioner, NDMemmap

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths


# TODO: make a mixin for source, distinct from clean command
class IndexTrain(IndexMixin, BaseModel):
    """Make a trained index base (inverted file type) from a dataset."""

    source: CliPositionalArg[DirectoryPath] = Field(
        description="HuggingFace dataset to train on."
    )
    dimensions: int | None = Field(  # matryoshka
        None,
        validation_alias=AliasChoices("d", "dimensions"),
        description=(
            "dimensions to truncate to when using Matryoshka "
            "embedding models, implies -N"
        ),
    )
    normalize: bool = Field(
        False,
        validation_alias=AliasChoices("N", "normalize"),
        description="use normalized embeddings when training index",
    )
    preprocess: str = Field(
        "OPQ96_384",
        validation_alias=AliasChoices("p", "preprocess"),
        description=(
            "preprocessing to use when training index, uses FAISS index factory syntax"
        ),
    )
    clusters: int | None = Field(
        None,
        validation_alias=AliasChoices("c", "clusters"),
        description="Number of IVF clusters",
    )

    _coerced_normalize: bool = PrivateAttr(False)
    _ivf_encoding: str = PrivateAttr()

    @model_validator(mode="after")
    def coerce_normalize(self) -> Self:
        if self.dimensions is not None and not self.normalize:
            self.normalize = True
            self._coerced_normalize = True
        return self

    @model_validator(mode="after")
    def match_preprocess(self) -> Self:
        if match := OPQ_PATTERN.match(self.preprocess):
            self._ivf_encoding = f"PQ{match[1]}"
            encoding_width = int(match[1])
            if encoding_width not in GPU_OPQ_WIDTHS:
                raise ValueError(f"OPQ width {encoding_width} is not valid")
        elif match := RR_PATTERN.match(self.preprocess):
            self._ivf_encoding = "SQ8"
            encoding_width = int(match[1])
        else:
            raise ValueError(f'preprocessing string "{self.preprocess}" is not valid')
        return self

    @override
    def cli_cmd(self) -> None:
        super().cli_cmd()

        dataset = load_dataset(self.source)

        if self.clusters is None:
            clusters = len(dataset) // TRAIN_SIZE_MULTIPLE
        else:
            clusters = self.clusters
        factory_string = f"{self.preprocess},IVF{clusters},{self._ivf_encoding}"
        train_size = TRAIN_SIZE_MULTIPLE * clusters

        shuffled = dataset.shuffle(seed=42)
        train = shuffled.take(train_size)

        train_memmap = _provision_memmap(
            train, self.dimensions, self.normalize, progress=self.progress
        )
        index = _train_index(train_memmap, factory_string)
        with del_on_exc([self.empty_index_path, self.untuned_params_path]):
            faiss.write_index(index, str(self.empty_index_path))
            save_params(self.untuned_params_path, self.dimensions, self.normalize, None)


def _provision_memmap(
    dataset: Dataset,
    dimensions: int | None,
    normalize: bool,
    progress: bool = False,
) -> NDMemmap[np.float32]:
    n = len(dataset)
    d = resolve_dimensions(dataset, dimensions)
    provisioner = MemmapProvisioner(dataset=dataset, shape=(n, d), normalize=normalize)
    return provisioner.provision(progress=progress)


def _train_index(
    train_memmap: NDMemmap[np.float32], factory_string: str
) -> faiss.Index:
    # doing a bit of testing seems to show that passing METRIC_L2 is superior to passing
    # METRIC_INNER_PRODUCT for the same factory string, even for normalized embeddings
    _, d = train_memmap.shape
    index: faiss.Index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)

    index = to_gpu(index)
    index.train(train_memmap)  # type: ignore # faiss class_wrappers.py
    index = to_cpu(index)

    return index
