from pathlib import Path
from typing import cast

from data.GMM_load import load_gmm
from lib.mvg import mvg_log_density
from lib.types import F64Array, F64Matrix, F64Tensor3D
from numpy import array, average, exp, load, log, save
from scipy.special import logsumexp

DATA_PATH = Path(__file__).parent / "data"

type GmmComponent = tuple[float, F64Array, F64Matrix]


def compute_log_joint(data: F64Matrix, gmm: list[GmmComponent]) -> F64Matrix:
    n_components = len(gmm)
    S = []
    weights = [w for w, _, _ in gmm]
    for _, mean, cov in gmm:
        S.append(mvg_log_density(data, mean, cov))
    return array(S).reshape((n_components, -1)) + log(weights).reshape((-1, 1))


def logpdf_gmm(data: F64Matrix, gmm: list[GmmComponent]) -> F64Array:
    return cast(F64Array, logsumexp(compute_log_joint(data, gmm), axis=0))


def compute_responsibilities(data: F64Matrix, gmm: list[GmmComponent]) -> F64Matrix:
    S = compute_log_joint(data, gmm)
    log_marginal = cast(F64Array, logsumexp(S, axis=0))
    return exp(S - log_marginal)


def update_gmm(data: F64Matrix, responsibilities: F64Matrix) -> list[GmmComponent]:
    gmm = []
    tot_fr_per_component = responsibilities.sum(axis=1)
    tot_samples = data.shape[1]

    for component in range(responsibilities.shape[0]):
        tot_fr = tot_fr_per_component[component]
        weight = tot_fr / tot_samples

        comp_responsibilities = responsibilities[component, :].ravel()
        mean = average(data, weights=comp_responsibilities, axis=1).ravel()

        cov_m = comp_responsibilities * data @ data.T / tot_fr - mean.reshape(
            (-1, 1)
        ) @ mean.reshape((1, -1))

        gmm.append((weight, mean, cov_m))

    return gmm


def train(
    data: F64Matrix, initialization: list[GmmComponent], stop_delta: float
) -> list[GmmComponent]:
    gmm = initialization
    ll_average = None
    while True:
        new_ll_avg = float(average(logpdf_gmm(data, gmm)))
        print(new_ll_avg)
        if ll_average is not None and new_ll_avg - ll_average < stop_delta:
            break
        ll_average = new_ll_avg
        responsibilities = compute_responsibilities(data, gmm)
        gmm = update_gmm(data, responsibilities)
    return gmm


def main():
    train_data_path = DATA_PATH / "GMM_data_4D.npy"
    gmm_path = DATA_PATH / "GMM_4D_3G_init.json"
    train_data = load(train_data_path)
    gmm = load_gmm(gmm_path)

    train(train_data, gmm, 1e-6)


if __name__ == "__main__":
    main()
