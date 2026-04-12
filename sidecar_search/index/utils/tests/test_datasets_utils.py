from itertools import batched
from pathlib import Path

import pytest
import torch
from datasets import Dataset, Features, List, Value

from ..datasets_utils import iter_tensors, load_dataset, resolve_dimensions

TEST_EMBEDDING_LEN = 3


def make_features(embedding_len: int) -> Features:
    return Features(
        {
            "id": Value("string"),
            "embedding": List(Value("float32"), length=embedding_len),
            "index": Value("int32"),
        }
    )


def make_empty_dataset(embedding_len: int) -> Dataset:
    features = make_features(embedding_len)
    return Dataset.from_dict({name: [] for name in features.to_dict()}, features)


class TestLoadDataset:
    @pytest.mark.parametrize("n_files", range(1, 3))
    def test_load(self, n_files: int, tmp_path: Path) -> None:
        features = make_features(TEST_EMBEDDING_LEN)

        expected_ids: list[str] = []
        expected_embeddings: list[list[float]] = []
        shard_features = features.copy()
        del shard_features["index"]  # our shards don't have this column
        for i in range(n_files):
            id_ = f"W{i}"
            embedding = [float(i)] * TEST_EMBEDDING_LEN

            mapping = {"id": [id_], "embedding": [embedding]}
            shard = Dataset.from_dict(mapping, shard_features)
            shard.to_parquet(tmp_path / f"data_{i:03}.parquet")

            expected_ids.append(id_)
            expected_embeddings.append(embedding)

        dataset = load_dataset(tmp_path)

        assert dataset.features == features
        assert dataset["id"] == expected_ids
        assert dataset["embedding"] == expected_embeddings

    def test_raise_on_no_parquet_files(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _ = load_dataset(tmp_path)


class TestResolveDimensions:
    def test_get_embedding_length(self) -> None:
        dataset = make_empty_dataset(TEST_EMBEDDING_LEN)
        assert resolve_dimensions(dataset, None) == TEST_EMBEDDING_LEN

    def test_pass_explicit_length(self) -> None:
        desired_len = 2
        dataset = make_empty_dataset(TEST_EMBEDDING_LEN)
        assert resolve_dimensions(dataset, desired_len) == desired_len

    def test_raise_on_variable_length(self) -> None:
        dataset = make_empty_dataset(-1)
        with pytest.raises(TypeError):
            _ = resolve_dimensions(dataset, None)

    def test_raise_on_missing_field(self) -> None:
        mapping = {"a": [0, 1, 2]}
        dataset = Dataset.from_dict(mapping)
        with pytest.raises(TypeError):
            _ = resolve_dimensions(dataset, None)

    def test_raise_on_incorrect_type(self) -> None:
        mapping = {"embedding": ["a", "b", "c"]}
        dataset = Dataset.from_dict(mapping)
        with pytest.raises(TypeError):
            _ = resolve_dimensions(dataset, None)


@pytest.mark.parametrize("dataset_len", range(3))
@pytest.mark.parametrize("batch_size", range(1, 3))
def test_iter_tensors(dataset_len: int, batch_size: int) -> None:
    # set up the underlying data
    dataset_range = range(dataset_len)
    ids = [f"W{i}" for i in dataset_range]
    embedding = [[float(i)] * TEST_EMBEDDING_LEN for i in dataset_range]
    index = list(dataset_range)
    mapping = {"id": ids, "embedding": embedding, "index": index}

    # cast that data to a Dataset
    features = make_features(TEST_EMBEDDING_LEN)
    dataset = Dataset.from_dict(mapping, features)

    # set up the expected output
    expected_index_batches = [
        torch.tensor(index_batch) for index_batch in batched(index, batch_size)
    ]
    expected_embedding_batches = [
        torch.tensor(embedding_batch)
        for embedding_batch in batched(embedding, batch_size)
    ]
    expected_batches = list(zip(expected_index_batches, expected_embedding_batches))

    batches = list(iter_tensors(dataset, batch_size=batch_size))

    assert all(
        (
            torch.equal(index_batch, expected_index_batch)
            and torch.equal(embedding_batch, expected_embedding_batch)
        )
        for (index_batch, embedding_batch), (
            expected_index_batch,
            expected_embedding_batch,
        ) in zip(batches, expected_batches)
    )
