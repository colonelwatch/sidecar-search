import pytest

from ..env_utils import bool_from_str, get_env_var

TEST_ENV = "SIDECARSEARCH_TEST"


class TestGetEnvVar:
    def test_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TEST_ENV, "var")
        assert get_env_var(TEST_ENV) == "var"

    def test_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TEST_ENV, raising=False)
        assert get_env_var(TEST_ENV) is None

    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TEST_ENV, raising=False)
        assert get_env_var(TEST_ENV, default="def") == "def"

    def test_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TEST_ENV, "42")
        assert get_env_var(TEST_ENV, int) == 42

    def test_default_not_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TEST_ENV, raising=False)
        sentinel = object()
        assert get_env_var(TEST_ENV, int, default=sentinel) is sentinel


class TestBoolFromStr:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0", False),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
        ],
    )
    def test_cast(self, value: str, expected: bool) -> None:
        assert bool_from_str(value) is expected

    def test_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = bool_from_str("var")
