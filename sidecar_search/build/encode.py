from typing import Callable, Generator, Iterable, Sequence

import torch
from sentence_transformers import SentenceTransformer

from sidecar_search.utils.gpu_utils import imap_multi_gpu, iunsqueeze

DocumentIdBatch = Sequence[tuple[str, str]]
DocumentEmbeddingBatch = tuple[Sequence[str], torch.Tensor]  # TODO: also convert to AoS


def get_model(
    model_name: str, bf16: bool, trust_remote_code: bool
) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch.bfloat16 if bf16 else torch.float16},
    )


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
    inputs: Iterable[DocumentIdBatch],
    model_factory: Callable[[], SentenceTransformer],
    *,
    tasks_per_gpu: int = 1,
) -> Generator[DocumentEmbeddingBatch, None, None]:
    models = [model_factory().to(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    def _encode(
        device: torch.device, batch: DocumentIdBatch
    ) -> tuple[list[str], torch.Tensor]:
        ids: list[str] = []
        documents: list[str] = []
        for id_, document in batch:
            ids.append(id_)
            documents.append(document)

        model = models[device.index]
        return ids, encode_faster(model, documents)

    batches = iunsqueeze(inputs)
    yield from imap_multi_gpu(batches, _encode, tasks_per_gpu=tasks_per_gpu)
