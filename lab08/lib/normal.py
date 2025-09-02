from numpy import exp, log, pi
from numpy.linalg import inv, slogdet

from .types import F64Array, F64Matrix


def log_normal_density(
    data: F64Matrix, mean: F64Matrix, covariance: F64Matrix
) -> F64Array:
    """Returns the log of the probability density function of the normal distribution identified by the parameters."""
    for x in [data, mean, covariance]:
        assert len(x.shape) == 2
    M: int = data.shape[0]
    for x in [mean, covariance]:
        assert x.shape[0] == M

    data_centered = data - mean
    return -0.5 * (
        M * log(2 * pi)
        + slogdet(covariance)[1]
        + ((data_centered.T @ inv(covariance)).T * data_centered).sum(axis=0)
    )


def normal_density(data: F64Matrix, mean: F64Matrix, covariance: F64Matrix) -> F64Array:
    return exp(log_normal_density(data, mean, covariance))
