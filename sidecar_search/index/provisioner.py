import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset
from datasets.fingerprint import Hasher

from sidecar_search.utils.cache_utils import get_cache_dir


class Provisioner[T](ABC):
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @abstractmethod
    def provision(self, progress: bool = False) -> T: ...

    def _compute_cache_path(self) -> Path:
        return get_cache_dir() / self._compute_cache_filename()

    @abstractmethod
    def _compute_cache_filename(self) -> str:
        return self._compute_cache_hash()

    def _compute_cache_hash(self) -> str:
        sorted_keys = sorted(self._kwargs.keys())
        hasher = Hasher()
        for k in sorted_keys:
            v = self._kwargs[k]
            if isinstance(v, Dataset):
                v = v._fingerprint
            if isinstance(v, Path):
                if not v.exists():
                    raise ValueError("input file does not exist")
                v = (v, os.path.getmtime(v))
            hasher.update(v)
        return hasher.hexdigest()
