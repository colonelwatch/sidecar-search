import faiss
import numpy as np
import numpy.typing as npt

from ..parameters import IndexParameters


def tune_index(
    index: faiss.Index,
    gt_queries: npt.NDArray[np.float32],
    gt_ids: npt.NDArray[np.int64],
    intersection: int | None,
    progress: bool = False,
) -> faiss.OperatingPoints:
    if len(gt_queries) != len(gt_ids):
        raise ValueError("gt_queries and gt_ids do not have matching lengths")

    n = len(gt_queries)

    # init with ground-truth IDs but not ground-truth distances because faiss doesn't
    # use them anyway (see faiss/AutoTune.cpp)
    if intersection is None:
        crit = faiss.OneRecallAtRCriterion(n, 1)
    else:
        crit = faiss.IntersectionCriterion(n, intersection)
    crit.set_groundtruth(None, gt_ids)  # type: ignore # faiss class_wrappers.py

    p_space = faiss.ParameterSpace()
    p_space.verbose = progress
    p_space.initialize(index)
    results = p_space.explore(index, gt_queries, crit)  # type: ignore # faiss class_wrappers.py
    assert isinstance(results, faiss.OperatingPoints), (
        "faiss violated documentation about return type"
    )

    return results


def serialize_operating_points(points: faiss.OperatingPoints) -> list[IndexParameters]:
    pareto_vector: faiss.OperatingPointVector = points.optimal_pts
    optimal_params: list[IndexParameters] = []
    for i in range(pareto_vector.size()):
        point: faiss.OperatingPoint = pareto_vector.at(i)
        params = IndexParameters(  # converts from ms to seconds
            recall=point.perf, exec_time=(0.001 * point.t), param_string=point.key
        )
        optimal_params.append(params)
    return optimal_params
