from typing import cast

from lib.mvg import mvg_log_density
from lib.types import F64Array, F64Matrix, F64Tensor3D
from numpy import log, vstack
from scipy.special import logsumexp


def logpdf_gmm(
    data: F64Matrix, means: F64Matrix, covs: list[F64Matrix], weights: F64Array
) -> F64Array:

    S = vstack(
        [mvg_log_density(data, means[:, i], covs[i]) for i in range(weights.size)]
    ) + log(weights).reshape((-1, 1))
    return cast(F64Array, logsumexp(S, axis=0))


def main():
    pass


if __name__ == "__main__":
    main()
