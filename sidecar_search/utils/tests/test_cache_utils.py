import gc
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from .. import cache_utils
from ..cache_utils import (
    PersistentCache,
    clean_persistent_cache,
    get_cache_dir,
    seal_persistent_cache,
)


# NOTE: should not be used on tests with calls to `.clean`
@pytest.fixture
def tmp_cache_dir_path(tmp_path: Path) -> Path:
    cache_dir_path = tmp_path / "cache"
    cache_dir_path.mkdir()
    return cache_dir_path


class TestPersistentCache:
    def test_get(self, tmp_cache_dir_path: Path) -> None:
        cache = PersistentCache(tmp_cache_dir_path)
        assert cache.get() == tmp_cache_dir_path

    def test_seal(self, tmp_path: Path) -> None:
        cache_dir_path = tmp_path / "cache"
        cache_dir_path.mkdir()

        cache = PersistentCache(cache_dir_path)
        cache.seal()

        dir_path = cache.get()

        # confirm a subdir was created, then test whether it auto-deletes
        assert dir_path.parent == cache_dir_path
        del cache
        gc.collect()

        assert cache_dir_path.exists()
        assert not dir_path.exists()

    def test_clean(self, tmp_path: Path) -> None:
        cache_dir_path = tmp_path / "cache"
        cache_dir_path.mkdir()
        (cache_dir_path / "data").touch()  # ensure non-empty cache dir

        cache = PersistentCache(cache_dir_path)
        cache.clean()

        assert not cache_dir_path.exists()

    def test_clean_no_error_on_not_exists(self, tmp_path: Path) -> None:
        cache_dir_path = tmp_path / "cache"
        cache = PersistentCache(cache_dir_path)
        cache.clean()

    def test_raise_on_clean_after_seal(self, tmp_path: Path) -> None:
        cache_dir_path = tmp_path / "cache"
        cache_dir_path.mkdir()

        cache = PersistentCache(cache_dir_path)
        cache.seal()

        dir_path = cache.get()

        with pytest.raises(RuntimeError):
            cache.clean()

        assert cache_dir_path.exists()
        assert dir_path.exists()

    def test_seal_idempotency(self, tmp_cache_dir_path: Path) -> None:
        cache = PersistentCache(tmp_cache_dir_path)
        cache.seal()
        dir_path_1 = cache.get()
        cache.seal()
        dir_path_2 = cache.get()
        assert dir_path_1 == dir_path_2


def test_get_cache_dir(monkeypatch: pytest.MonkeyPatch):
    path_mock = MagicMock()
    mock = MagicMock()
    mock.get.return_value = path_mock
    monkeypatch.setattr(cache_utils, "cache", mock)

    ret = get_cache_dir()

    mock.get.assert_called_once()
    assert ret is path_mock


def test_seal_persistence_cache(monkeypatch: pytest.MonkeyPatch):
    mock = MagicMock()
    monkeypatch.setattr(cache_utils, "cache", mock)
    seal_persistent_cache()
    mock.seal.assert_called_once()


def test_clean_persistence_cache(monkeypatch: pytest.MonkeyPatch):
    mock = MagicMock()
    monkeypatch.setattr(cache_utils, "cache", mock)
    clean_persistent_cache()
    mock.clean.assert_called_once()
