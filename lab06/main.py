from typing import cast

from numpy import (
    average,
    cov,
    dtype,
    exp,
    eye,
    float64,
    full,
    load,
    log,
    ndarray,
    pi,
    stack,
    uint8,
    unique,
)
from numpy.linalg import inv, slogdet
from numpy.random import permutation, seed
from scipy.special import logsumexp
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type F64Array = ndarray[tuple[int], dtype[float64]]
type U8Array = ndarray[tuple[int], dtype[uint8]]


def load_data(binary=False) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    load_res = cast(dict[str, ndarray], load_iris())
    data, target = load_res["data"].T, load_res["target"]
    if binary:
        data = data[:, target != 0]
        target = target[target != 0]
    return split_train_test(data, target)


def split_train_test(
    data: F64Matrix, target: U8Array
) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    tot_samples = data.shape[1]
    train_count = int(tot_samples * 2 / 3)
    seed(0)
    idx = permutation(data.shape[1])
    train_idx, test_idx = idx[:train_count], idx[train_count:]
    return data[:, train_idx], target[train_idx], data[:, test_idx], target[test_idx]


def log_normal_density(
    data: F64Matrix, mean: F64Matrix, covariance: F64Matrix
) -> F64Array:
    n_dim: int = data.shape[0]

    data_centered = data - mean
    return -0.5 * (
        n_dim * log(2 * pi)
        + slogdet(covariance)[1]
        + ((data_centered.T @ inv(covariance)).T * data_centered).sum(axis=0)
    )


def normal_density(data: F64Matrix, mean: F64Matrix, covariance: F64Matrix) -> F64Array:
    return exp(log_normal_density(data, mean, covariance))


def mvg_classifier():
    train_data, train_target, test_data, test_target = load_data()
    unique_targets = unique(train_target)
    n_dim, n_cls = train_target.shape[0], len(unique_targets)

    cls_means: list[F64Matrix] = [
        train_data[:, train_target == t].mean(axis=1).reshape(n_dim, 1)
        for t in unique_targets
    ]
    cls_covs: list[F64Matrix] = [
        cov(train_data[:, train_target == t], bias=True, dtype=float64)
        for t in unique_targets
    ]

    cls_likelyhood: F64Matrix = stack(
        [normal_density(test_data, m, c) for m, c in zip(cls_means, cls_covs)]
    )

    cls_prior: F64Matrix = full(n_cls, 1 / n_cls, dtype=float64).reshape(n_cls, 1)

    cls_joint = cls_likelyhood * cls_prior
    cls_posterior = normalize(cls_joint, axis=0, norm="l1")
    print(cls_posterior[:, :3])
    predicted_target = cls_posterior.argmax(axis=0)
    err_rate = (predicted_target != test_target).sum() / test_target.size
    print(f"{err_rate=}")


def mvg_log_classifier(naive=False, tied_cov=False):
    train_data, train_target, test_data, test_target = load_data()
    unique_targets = unique(train_target)
    n_dim, n_cls = train_data.shape[0], len(unique_targets)

    cls_means = [
        train_data[:, train_target == t].mean(axis=1).reshape(-1, 1)
        for t in unique_targets
    ]
    cls_covs = [
        cov(train_data[:, train_target == t], bias=True) for t in unique_targets
    ]

    if naive:
        cls_covs = [c * eye(n_dim, n_dim) for c in cls_covs]

    if tied_cov:
        within_class_cov = average(
            stack(cls_covs), axis=0, weights=unique(train_target, return_counts=True)[1]
        )
        cls_covs = [within_class_cov for _ in range(n_cls)]
        print(within_class_cov)

    cls_conditional_ll = stack(
        [
            log_normal_density(test_data, cls_means[t], cls_covs[t])
            for t in unique_targets
        ]
    )
    log_prior = log(normalize(full((n_cls, 1), 1), norm="l1", axis=0))
    log_joint = cls_conditional_ll + log_prior
    log_marginal: F64Matrix = cast(F64Array, logsumexp(log_joint, axis=0)).reshape(
        1, -1
    )
    log_posterior = log_joint - log_marginal
    posterior = exp(log_posterior)
    predicted = posterior.argmax(axis=0)
    err_rate = (predicted != test_target).sum() / test_target.size
    print(f"{err_rate=}")


def main():
    pass


if __name__ == "__main__":
    main()
