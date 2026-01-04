from typing import Generator, Iterable

import torch
from sentence_transformers import SentenceTransformer

from sidecar_search.utils.gpu_utils import imap, iunsqueeze, iunzip

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


def encode_pipelined(
    batches: Iterable[DocumentIdBatch],
    model: SentenceTransformer,
    n_tasks: int,
) -> Generator[DocumentEmbeddingBatch, None, None]:
    ids_batches, documents_batches = iunzip(batches, 2)
    documents_batches = iunsqueeze(documents_batches)
    embeddings_batches = imap(
        documents_batches, lambda x: encode_faster(model, x), n_tasks
    )
    batches_out = zip(ids_batches, embeddings_batches)
    for ids_batch, embeddings_batch in batches_out:
        yield ids_batch, embeddings_batch
