from numpy import average, dtype, ones, uint8
from sklearn.metrics import confusion_matrix

from .types import F64Array, F64Matrix, U8Array, U64Matrix


def dcf(conf_m: U64Matrix, cost: F64Matrix, prior: F64Array, normalized=True) -> float:
    assert 2 == conf_m.ndim == cost.ndim
    assert (
        conf_m.shape[0]
        == conf_m.shape[1]
        == cost.shape[0]
        == cost.shape[1]
        == prior.size
    )
    dcf_u = average((conf_m * cost).sum(axis=0) / conf_m.sum(axis=0), weights=prior)
    if not normalized:
        return dcf_u
    dcf_normalizer = (cost @ prior.reshape(-1, 1)).min()
    return dcf_u / dcf_normalizer


def dcf_min_bin(
    scores: F64Matrix,
    target: U8Array,
    cost: F64Matrix,
    prior: F64Array,
    normalized=True,
) -> float:
    conf_m = confusion_matrix(ones(target.size, dtype=uint8), target).T
    dcf_min = dcf(conf_m, cost, prior, normalized)
    for t in target[scores.argsort()]:
        conf_m[1, t] -= 1
        conf_m[0, t] += 1
        dcf_min = min(dcf_min, dcf(conf_m, cost, prior, normalized))
    return dcf_min
