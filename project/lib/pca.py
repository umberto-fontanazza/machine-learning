from numpy import cov
from scipy.linalg import eigh

from .types import F64Matrix


def get_pca_lt(data: F64Matrix, m: int) -> F64Matrix:
    """Returns the linear transformation matrix P reducing the input space to m dimensions"""
    assert len(data.shape) == 2
    assert data.shape[0] > m
    _, U = eigh(cov(data, bias=True))
    return U[:, ::-1][:, :m]
