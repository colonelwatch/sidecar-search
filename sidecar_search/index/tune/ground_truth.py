from itertools import accumulate, tee
from typing import TypedDict, Unpack

import torch
from datasets import Dataset
from tqdm import tqdm

from sidecar_search.utils.gpu_utils import imap, imap_multi_gpu

from ..provisioner import Provisioner
from ..utils.datasets_utils import iter_tensors


class GroundTruthKwargs(TypedDict):
    dataset: Dataset
    queries: Dataset
    do_inner_product_search: bool
    k: int


class GroundTruthBuilder:
    def __init__(self, **kwargs: Unpack[GroundTruthKwargs]) -> None:
        dataset = kwargs["dataset"]
        queries = kwargs["queries"]
        do_inner_product_search = kwargs["do_inner_product_search"]
        k = kwargs["k"]

        self._dataset = dataset
        self._queries = queries
        self._do_inner_product_search = do_inner_product_search
        self._k = k

        # get query embeddings and IDs, with a local copy for each GPU
        # TODO: calling _getitem is a workaround, bypassing the Column class
        #       used since datasets 4.0.0, but this should be rewritten sometime
        with queries.formatted_as("torch", columns=["embedding", "index"]):
            q_embeddings: torch.Tensor = queries._getitem("embedding")  # type: ignore
            q_ids: torch.Tensor = queries._getitem("index")  # type: ignore

        if do_inner_product_search:
            q_embeddings = torch.nn.functional.normalize(q_embeddings)
        n_devices = torch.cuda.device_count()
        self._q_embeddings = q_embeddings
        self._q_ids = q_ids
        self._n_devices = n_devices
        self._q_embeddings_copy = [
            q_embeddings.to(f"cuda:{i}") for i in range(n_devices)
        ]
        self._q_ids_copy = [q_ids.to(f"cuda:{i}") for i in range(n_devices)]

    def build(self, progress: bool = False) -> Dataset:
        # initialize the top k
        n_q, _ = self._q_embeddings.shape
        shape = (n_q, self._k)
        gt_ids = torch.full(shape, -1, dtype=torch.int32).cuda()
        if self._do_inner_product_search:
            gt_scores = torch.zeros(shape, dtype=torch.float32).cuda()
        else:
            gt_scores = torch.full(shape, torch.inf, dtype=torch.float32).cuda()

        with tqdm(
            desc="make_ground_truth",
            total=len(self._dataset),
            disable=(not progress),
        ) as counter:
            batches = iter_tensors(self._dataset)
            batches, batches_copy = tee(batches, 2)
            lengths = imap(batches_copy, self._get_length, 0)
            batches = imap_multi_gpu(batches, self._local_topk)
            batches = accumulate(
                batches, self._reduce_topk, initial=(gt_ids, gt_scores)
            )
            batches = zip(lengths, batches)
            for length, (gt_ids, _) in batches:
                counter.update(length)

        gt_ids = gt_ids.cpu()

        ground_truth = Dataset.from_dict(
            {
                "embedding": self._q_embeddings,
                "gt_ids": gt_ids.numpy(),
            }
        )
        return ground_truth

    def _get_length(self, ids: torch.Tensor, _: torch.Tensor) -> int:
        return len(ids)

    # NOTE: ground truth is computed with the full embedding length
    def _local_topk(
        self, device: torch.device, ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # send to GPU asynchronously
        embeddings = embeddings.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)

        # acquire device copy of queries
        q_ids = self._q_ids_copy[device.index]
        q_embeddings = self._q_embeddings_copy[device.index]

        # don't consider the queries themselves as possible ground truth
        not_in_queries = torch.isin(ids, q_ids, invert=True)
        embeddings = embeddings[not_in_queries]
        ids = ids[not_in_queries]

        if self._do_inner_product_search:
            # ensure that the vectors are unit-length
            embeddings = torch.nn.functional.normalize(embeddings)

            # becomes a matmult for multiple data
            scores = q_embeddings @ embeddings.T
        else:
            # prefer direct calc over following the quadratic form with matmult
            scores = torch.cdist(
                q_embeddings, embeddings, compute_mode="donot_use_mm_for_euclid_dist"
            )

        # only yield k from this batch, in the extreme this k replaces all running k
        top_scores, argtop = torch.topk(
            scores, self._k, dim=1, largest=self._do_inner_product_search
        )
        top_ids = ids[argtop.flatten()].reshape(argtop.shape)

        if self._n_devices > 1:
            return top_ids.cpu(), top_scores.cpu()
        else:
            return top_ids, top_scores  # reduce step is on this GPU

    def _reduce_topk(
        self,
        gt: tuple[torch.Tensor, torch.Tensor],
        batch_top: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_ids, gt_scores = gt
        batch_ids, batch_scores = batch_top

        batch_ids = batch_ids.cuda(non_blocking=True)
        batch_scores = batch_scores.cuda(non_blocking=True)

        # update the top k for each query
        gt_scores = torch.hstack((gt_scores, batch_scores))
        gt_ids = torch.hstack((gt_ids, batch_ids))
        gt_scores, argtop = torch.topk(
            gt_scores, self._k, dim=1, largest=self._do_inner_product_search
        )
        gt_ids = torch.gather(gt_ids, 1, argtop)

        return gt_ids, gt_scores


class GroundTruthProvisioner(Provisioner[Dataset]):
    def __init__(self, **kwargs: Unpack[GroundTruthKwargs]) -> None:
        super().__init__(**kwargs)

    def provision(self, progress: bool = False) -> Dataset:
        cache_path = self._compute_cache_path()
        if cache_path.exists():
            return Dataset.load_from_disk(cache_path)
        builder = GroundTruthBuilder(**self._kwargs)
        ground_truth = builder.build(progress=progress)
        ground_truth.save_to_disk(cache_path)
        return ground_truth

    def _compute_cache_filename(self) -> str:
        return f"gt_{self._compute_cache_hash()}"
