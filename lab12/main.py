from pathlib import Path
from typing import cast

from data.GMM_load import load_gmm
from lib.mvg import mvg_log_density
from lib.types import F64Array, F64Matrix
from numpy import array, average, cov, diag, exp, load, log
from numpy.linalg import svd
from scipy.special import logsumexp
from solution.gmm import save_gmm

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


def update_gmm(
    data: F64Matrix, responsibilities: F64Matrix, min_eig: float | None
) -> list[GmmComponent]:
    gmm = []
    tot_fr_per_component = responsibilities.sum(axis=1)
    tot_samples = data.shape[1]

    for component in range(responsibilities.shape[0]):
        tot_fr = tot_fr_per_component[component]
        weight = tot_fr / tot_samples

        comp_responsibilities = responsibilities[component, :].ravel()
        mean = average(data, weights=comp_responsibilities, axis=1).ravel()

        cov_m = cov(
            data - mean.reshape((-1, 1)), bias=True, aweights=comp_responsibilities
        )

        gmm.append(
            (weight, mean, constrain_cov_m(cov_m, min_eig) if min_eig else cov_m)
        )

    return gmm


def train_em(
    data: F64Matrix,
    initialization: list[GmmComponent],
    stop_delta: float,
    min_eig: float | None,
) -> list[GmmComponent]:
    gmm = initialization
    ll_average = None
    while True:
        new_ll_avg = float(average(logpdf_gmm(data, gmm)))
        if ll_average is not None and new_ll_avg - ll_average < stop_delta:
            break
        ll_average = new_ll_avg
        responsibilities = compute_responsibilities(data, gmm)
        gmm = update_gmm(data, responsibilities, min_eig)
    return gmm


def compute_displacement_vector(cov_m: F64Matrix, alpha: float) -> F64Array:
    U, s, _ = svd(cov_m)
    return (U[:, 0:1] * s[0] ** 0.5 * alpha).ravel()


def train_lbg(gmm: list[GmmComponent], alpha: float) -> list[GmmComponent]:
    splitted = []
    for weight, mean, cov_m in gmm:
        d = compute_displacement_vector(cov_m, alpha)
        splitted.append((weight / 2, mean - d, cov_m))
        splitted.append((weight / 2, mean + d, cov_m))
    return splitted


from numpy.linalg import svd


def constrain_cov_m(cov_m: F64Matrix, min_eig: float) -> F64Matrix:
    u, c, _ = svd(cov_m)
    c[c < min_eig] = min_eig
    assert c.ndim == 1
    return u @ diag(c) @ u.T


def train_gmm(
    data: F64Matrix,
    alpha: float,
    stop_delta: float,
    target_components: int,
    min_eig: float | None = None,
) -> list[GmmComponent]:
    cov_m = cast(F64Matrix, cov(data, bias=True))
    if min_eig:
        cov_m = constrain_cov_m(cov_m, min_eig)
    gmm = cast(list[GmmComponent], [(1, data.mean(axis=1), cov_m)])
    # assert target_components is a power of 2
    while len(gmm) < target_components:
        gmm = train_lbg(gmm, alpha)
        gmm = train_em(data, gmm, stop_delta, min_eig)
    return gmm


def main():
    train_data_path = DATA_PATH / "GMM_data_4D.npy"
    train_data = load(train_data_path)
    # gmm_path = DATA_PATH / "GMM_4D_3G_init.json"
    # gmm_path = DATA_PATH / "GMM_4D_4G_EM_LBG.json"
    # gmm = load_gmm(gmm_path)
    stop_delta = 1e-6
    alpha = 0.1
    target_components = 4
    my_gmm = train_gmm(train_data, alpha, stop_delta, target_components)
    save_gmm(my_gmm, DATA_PATH / "my_gmm.json")


if __name__ == "__main__":
    main()
