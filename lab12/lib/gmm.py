from typing import cast

from lib.mvg import mvg_log_density
from lib.types import F64Array, F64Matrix, GmmComponent
from numpy import array, average, cov, diag, exp, eye, log, zeros
from numpy.linalg import svd
from scipy.special import logsumexp


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
    data: F64Matrix,
    responsibilities: F64Matrix,
    min_eig: float | None,
    diag: bool,
    tied: bool,
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

        gmm.append((weight, mean, cov_m))

    if tied:
        cov_m = zeros((data.shape[0], data.shape[0]))
        for weight, _, c in gmm:
            cov_m += weight * c
        gmm = [(w, m, cov_m) for w, m, _ in gmm]

    if diag:
        gmm = [(w, m, c * eye(c.shape[0], c.shape[1])) for w, m, c in gmm]

    if min_eig is not None:
        gmm = [(w, m, constrain_cov_m(c, min_eig)) for w, m, c in gmm]

    return gmm


def train_em(
    data: F64Matrix,
    initialization: list[GmmComponent],
    stop_delta: float,
    min_eig: float | None,
    diag: bool,
    tied: bool,
) -> list[GmmComponent]:
    gmm = initialization
    ll_average = None
    while True:
        new_ll_avg = float(average(logpdf_gmm(data, gmm)))
        if ll_average is not None and new_ll_avg - ll_average < stop_delta:
            break
        ll_average = new_ll_avg
        responsibilities = compute_responsibilities(data, gmm)
        gmm = update_gmm(data, responsibilities, min_eig, diag, tied)
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
    diag: bool = False,
    tied: bool = False,
) -> list[GmmComponent]:
    cov_m = cast(F64Matrix, cov(data, bias=True))
    if min_eig:
        cov_m = constrain_cov_m(cov_m, min_eig)
    gmm = cast(list[GmmComponent], [(1, data.mean(axis=1), cov_m)])
    # assert target_components is a power of 2
    while len(gmm) < target_components:
        gmm = train_lbg(gmm, alpha)
        gmm = train_em(data, gmm, stop_delta, min_eig, diag, tied)
    return gmm
