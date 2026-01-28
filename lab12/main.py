from pathlib import Path
from typing import cast

from data.GMM_load import load_gmm
from lib.mvg import mvg_log_density
from lib.types import F64Array, F64Matrix, F64Tensor3D
from numpy import array, load, log, save, vstack
from scipy.special import logsumexp

DATA_PATH = Path(__file__).parent / "data"


def logpdf_gmm(
    data: F64Matrix, means: F64Matrix, covs: list[F64Matrix], weights: F64Array
) -> F64Array:

    S = vstack(
        [mvg_log_density(data, means[:, i], covs[i]) for i in range(weights.size)]
    ) + log(weights).reshape((-1, 1))
    return cast(F64Array, logsumexp(S, axis=0))


def logpdf_gmm2(
    data: F64Matrix, gmm: list[tuple[float, F64Array, F64Matrix]]
) -> F64Array:
    n_components = len(gmm)
    S = []
    weights = [w for w, _, _ in gmm]
    for _, mean, cov in gmm:
        S.append(mvg_log_density(data, mean, cov))
    S = array(S).reshape((n_components, -1)) + log(weights).reshape((-1, 1))
    return cast(F64Array, logsumexp(S, axis=0))


def main():
    train_data_path = DATA_PATH / "GMM_data_4D.npy"
    gmm_path = DATA_PATH / "GMM_4D_3G_init.json"
    train_data = load(train_data_path)
    gmm = load_gmm(gmm_path)
    weights, means, covs = zip(*gmm)
    weights, means, covs = (
        array(weights),
        array([m.ravel() for m in means]).T,
        list(covs),
    )
    computed = logpdf_gmm(train_data, means, covs, weights)
    computed = computed.reshape((1, -1))
    save(DATA_PATH / "my_res.npy", computed)


if __name__ == "__main__":
    main()
