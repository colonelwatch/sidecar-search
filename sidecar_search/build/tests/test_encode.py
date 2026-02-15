from itertools import chain, cycle
from typing import Any, cast
from unittest.mock import ANY, MagicMock, create_autospec

import pytest
import torch
from sentence_transformers import SentenceTransformer

from sidecar_search.build.encode import PipelinedEncoder, encode_faster, get_model
from sidecar_search.utils.gpu_utils import imap_multi_gpu

EMBEDDING_KEY = "sentence_embedding"


@pytest.mark.parametrize("bf16", (True, False))
def test_get_model(monkeypatch: pytest.MonkeyPatch, bf16: bool) -> None:
    mock_cls: MagicMock = create_autospec(  # duck-typing as MagicMock
        spec=SentenceTransformer
    )
    monkeypatch.setattr("sidecar_search.build.encode.SentenceTransformer", mock_cls)

    model_name = "asdfgh"
    trust_remote_code = True
    _ = get_model(model_name, bf16, trust_remote_code)

    expected_model_kwargs = {"torch_dtype": torch.bfloat16 if bf16 else torch.float16}
    mock_cls.assert_called_once_with(
        model_name,
        trust_remote_code=trust_remote_code,
        model_kwargs=expected_model_kwargs,
    )


def make_mock_model(
    tokenize_return_value: dict[str, torch.Tensor] | None = None,
    forward_return_value: dict[str, torch.Tensor] | None = None,
) -> MagicMock:
    mock_model = MagicMock(spec=SentenceTransformer)

    mock_model.device = "cpu"
    mock_model.to.return_value = mock_model

    if tokenize_return_value is not None:
        mock_model.tokenize.return_value = tokenize_return_value

    if forward_return_value is not None:
        mock_model.forward.return_value = forward_return_value

    return mock_model


def test_encode_faster() -> None:
    sentences = ["sentence"]
    features = {"key": torch.arange(10).reshape(1, -1)}
    embeddings = {EMBEDDING_KEY: torch.arange(10).reshape(1, -1)}

    mock_model = make_mock_model(features, embeddings)

    results = encode_faster(mock_model, sentences)

    mock_model.tokenize.assert_called_once_with(sentences)
    mock_model.forward.assert_called_once_with(features)
    assert torch.equal(results, embeddings["sentence_embedding"])


@pytest.mark.usefixtures("mock_gpu_env")
class TestPipelinedEncoder:
    def test_tasks_per_gpu_arg_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_imap = cast(  # duck-typing as MagicMock
            MagicMock, create_autospec(spec=imap_multi_gpu)
        )
        mock_imap.side_effect = lambda *args, **kwargs: (x for x in iter([]))
        monkeypatch.setattr("sidecar_search.build.encode.imap_multi_gpu", mock_imap)

        tasks_per_gpu = 10
        encoder = PipelinedEncoder(make_mock_model, tasks_per_gpu=tasks_per_gpu)
        batches = iter([])
        _ = list(encoder.encode(batches))

        mock_imap.assert_called_once_with(ANY, ANY, tasks_per_gpu=tasks_per_gpu)

    def test_round_robin_encode(self) -> None:
        counter = 0
        mock_models: list[MagicMock] = []

        def provision_mock_model() -> Any:
            nonlocal counter
            embedding = torch.tensor([[counter]])
            mock_model = make_mock_model(
                forward_return_value={"sentence_embedding": embedding}
            )
            mock_models.append(mock_model)
            counter += 1
            return mock_model

        n_gpus = torch.cuda.device_count()
        cycles = list(chain(range(n_gpus), range(n_gpus)))

        encoder = PipelinedEncoder(provision_mock_model)
        batches = ([(f"id{i}", f"document{i}")] for i in cycles)

        results = list(encoder.encode(batches))

        for i, mock_model in zip(cycles, cycle(mock_models)):
            mock_model.to.assert_called_with(f"cuda:{i}")
            mock_model.tokenize.assert_called_with([f"document{i}"])

        for i, batch in zip(cycle(range(n_gpus)), results):
            assert batch == [(f"id{i}", torch.tensor([[i]]))]

    def test_empty_iterator(self) -> None:
        encoder = PipelinedEncoder(make_mock_model)
        batches = iter([])
        results = list(encoder.encode(batches))
        assert results == []

    def test_empty_batch(self) -> None:
        encoder = PipelinedEncoder(make_mock_model)
        batches = iter([[]])
        results = list(encoder.encode(batches))
        assert results == [[]]
