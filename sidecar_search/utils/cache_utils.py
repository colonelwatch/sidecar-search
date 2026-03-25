from pathlib import Path
from shutil import rmtree
from sys import stderr
from tempfile import TemporaryDirectory
from threading import Lock

from datasets import Dataset, disable_caching
from datasets.config import HF_DATASETS_CACHE

from .env_utils import CACHE


class PersistentCache:
    def __init__(self, cache_dir_path: Path) -> None:
        self._cache_dir_path = cache_dir_path
        self._temp_dir: TemporaryDirectory | None = None

        self._lock = Lock()
        self._dir_path = cache_dir_path

    def get(self) -> Path:
        with self._lock:
            return self._get_nolock()

    def seal(self) -> None:
        with self._lock:
            # replace with a TemporaryDirectory (cleanup upon interpreter exit)
            dir_path = self._get_nolock()
            if dir_path != self._cache_dir_path:
                return
            temp_dir = TemporaryDirectory(dir=str(dir_path))

            self._temp_dir = temp_dir
            self._dir_path = Path(temp_dir.name)

    def clean(self) -> None:
        with self._lock:
            cache_dir_path = self._cache_dir_path
            if cache_dir_path != self._dir_path:
                raise RuntimeError("Cleaned persistent cache after sealing it")
            if not cache_dir_path.exists():
                return
            rmtree(cache_dir_path)

    def _get_nolock(self) -> Path:
        dir_path = self._dir_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


cache = PersistentCache(CACHE)


def get_cache_dir() -> Path:
    return cache.get()


def seal_persistent_cache() -> None:
    cache.seal()


def clean_persistent_cache() -> None:
    cache.clean()


def seal_hf_cache() -> None:
    disable_caching()


def clean_hf_cache(dataset: Dataset):
    # get cache directory path by following the path to an individual cache file
    file_0_path = Path(dataset.cache_files[0]["filename"])
    del dataset

    # parts[0] -> dataset ("parquet" by default)
    # parts[1] -> cache (pseudorandom, seeded with stuff like file metadata)
    # since this is a low-level detail, sanity-check the above facts for change
    file_0_path_rel = file_0_path.relative_to(HF_DATASETS_CACHE)
    dataset_name = file_0_path_rel.parts[0]
    cache_name = file_0_path_rel.parts[1]
    if not (dataset_name == "parquet" and "default-" in cache_name):
        print("error: path integrity check failed", file=stderr)
        return 1

    # remove the cache directory
    hf_cache_dir = HF_DATASETS_CACHE / dataset_name / cache_name
    rmtree(hf_cache_dir)

    # remove its associated lock
    for lock in HF_DATASETS_CACHE.iterdir():
        if not lock.suffix == ".lock":
            continue
        if cache_name in str(lock):
            lock.unlink()
