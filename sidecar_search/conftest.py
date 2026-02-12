import pytest


class MockTorchDevice:
    def __init__(self, device: str) -> None:
        type_, index_str = device.split(":", maxsplit=2)
        self.type = type_
        self.index = int(index_str)


@pytest.fixture()
def mock_gpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr("torch.device", MockTorchDevice)
