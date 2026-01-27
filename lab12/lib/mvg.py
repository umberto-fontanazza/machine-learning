from lib.types import F64Array, F64Matrix
from numpy import log
from numpy.linalg import inv, slogdet
from scipy.constants import pi


def mvg_log_density(data: F64Matrix, mean: F64Array, cov: F64Matrix) -> F64Array:
    """Computes the logarithm of the density of a multivariate gaussian distribution."""
    D: int = data.shape[0]  # dimensionality of samples
    centered_data = data - mean.reshape((-1, 1))
    return (
        -(
            D * log(2 * pi)
            + slogdet(cov)[1]
            + ((centered_data.T @ inv(cov)).T * centered_data).sum(axis=0)
        )
        / 2
    )
