from typing import Callable, Generator, Iterable

import torch
from sentence_transformers import SentenceTransformer

from sidecar_search.utils.gpu_utils import imap_multi_gpu, iunsqueeze, iunzip

DocumentIdBatch = tuple[list[str], list[str]]
DocumentEmbeddingBatch = tuple[list[str], torch.Tensor]


# built from SentenceTransformer.encode but with non-blocking CPU-to-GPU transfers
def encode_faster(
    model: SentenceTransformer,
    sentences: list[str],
) -> torch.Tensor:
    model.eval()

    # Tokenize (which yields a dict) then do a non-blocking transfer
    features = {
        k: v.to(model.device, non_blocking=True)
        for k, v in model.tokenize(sentences).items()
    }

    with torch.no_grad():
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]

    return embeddings.cpu()


# TODO: make a class out of this
def encode_pipelined(
    batches: Iterable[DocumentIdBatch],
    model_factory: Callable[[], SentenceTransformer],
    *,
    tasks_per_gpu: int = 1,
) -> Generator[DocumentEmbeddingBatch, None, None]:
    models = [model_factory().to(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    def _encode(device: torch.device, sentences: list[str]) -> torch.Tensor:
        model = models[device.index]
        return encode_faster(model, sentences)

    ids_batches, documents_batches = iunzip(batches, 2)
    documents_batches = iunsqueeze(documents_batches)
    embeddings_batches = imap_multi_gpu(
        documents_batches, _encode, tasks_per_gpu=tasks_per_gpu
    )
    batches_out = zip(ids_batches, embeddings_batches)
    for ids_batch, embeddings_batch in batches_out:
        yield ids_batch, embeddings_batch
