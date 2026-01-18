import re
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from datasets import Dataset

from sidecar_search.args_base import SubcommandArgsBase
from sidecar_search.utils.contextmanager_utils import del_on_exc

from ..args import IndexSharedArgsMixin
from ..parameters import save_params
from ..utils.datasets_utils import resolve_dimensions
from ..utils.faiss_utils import to_cpu, to_gpu
from .memmap import MemmapProvisioner, NDMemmap

TRAIN_SIZE_MULTIPLE = 50  # x clusters = train size recommended by FAISS folks
OPQ_PATTERN = re.compile(r"OPQ([0-9]+)(?:_([0-9]+))?")
RR_PATTERN = re.compile(r"(?:PCAR|RR)([0-9]+)")  # RR <==> PCAR without the PCA
GPU_OPQ_WIDTHS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96]  # GPU widths


# TODO: make a mixin for source, distinct from clean command
@dataclass
class IndexTrainArgs(
    IndexSharedArgsMixin, SubcommandArgsBase[Literal["index"], Literal["train"]]
):
    source: Path
    dimensions: int | None  # matryoshka
    normalize: bool  # also if normalize_d_
    preprocess: str
    clusters: int | None

    # not args
    ivf_encoding: str = field(init=False, compare=False)
    encoding_width: int = field(init=False, compare=False)

    @classmethod
    def configure_parser(cls, parser: ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument("source", type=Path)
        parser.add_argument("-d", "--dimensions", default=None, type=int)
        parser.add_argument("-N", "--normalize", action="store_true")
        parser.add_argument("-p", "--preprocess", default="OPQ96_384")
        parser.add_argument("-c", "--clusters", default=None, type=int)

    def __post_init__(self):
        super().__post_init__()

        if not self.source.exists():
            raise ValueError(f'source path "{self.source}" does not exist')

        if self.dimensions is not None and not self.normalize:
            self.normalize = True
            warnings.warn("inferring --normalize from --dimension")

        if match := OPQ_PATTERN.match(self.preprocess):
            self.ivf_encoding = f"PQ{match[1]}"
            self.encoding_width = int(match[1])
            if self.encoding_width not in GPU_OPQ_WIDTHS:
                raise ValueError(f"OPQ width {self.encoding_width} is not valid")
        elif match := RR_PATTERN.match(self.preprocess):
            self.ivf_encoding = "SQ8"
            self.encoding_width = int(match[1])
        else:
            raise ValueError(f'preprocessing string "{self.preprocess}" is not valid')


def provision_memmap(dataset: Dataset, args: IndexTrainArgs) -> NDMemmap[np.float32]:
    n = len(dataset)
    d = resolve_dimensions(dataset, args.dimensions)
    provisioner = MemmapProvisioner(
        dataset=dataset, shape=(n, d), normalize=args.normalize
    )
    return provisioner.provision(progress=args.progress)


def train_index(
    train: Dataset, factory_string: str, args: IndexTrainArgs
) -> faiss.Index:
    train_memmap = provision_memmap(train, args)

    # doing a bit of testing seems to show that passing METRIC_L2 is superior to passing
    # METRIC_INNER_PRODUCT for the same factory string, even for normalized embeddings
    _, d = train_memmap.shape
    index: faiss.Index = faiss.index_factory(d, factory_string, faiss.METRIC_L2)

    index = to_gpu(index)
    index.train(train_memmap)  # type: ignore # faiss class_wrappers.py
    index = to_cpu(index)

    return index


def ensure_trained(dataset: Dataset, args: IndexTrainArgs):
    if args.clusters is None:
        clusters = len(dataset) // TRAIN_SIZE_MULTIPLE
    else:
        clusters = args.clusters
    factory_string = f"{args.preprocess},IVF{clusters},{args.ivf_encoding}"
    train_size = TRAIN_SIZE_MULTIPLE * clusters

    shuffled = dataset.shuffle(seed=42)
    train = shuffled.take(train_size)

    index = train_index(train, factory_string, args)
    with del_on_exc([args.empty_index_path, args.untuned_params_path]):
        faiss.write_index(index, str(args.empty_index_path))
        save_params(args.untuned_params_path, args.dimensions, args.normalize, None)
