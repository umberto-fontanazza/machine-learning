from numpy import array, float64

from .types import F64Matrix


def cost_from_fn_fp(cost_fn: float, cost_fp: float) -> F64Matrix:
    return array([[0, cost_fn], [cost_fp, 0]], dtype=float64)
