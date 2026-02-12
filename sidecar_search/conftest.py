import pytest


@pytest.fixture()
def mock_gpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
