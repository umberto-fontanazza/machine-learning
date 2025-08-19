from typing import cast

from matplotlib.pyplot import hist, legend, scatter, show
from numpy import (
    array,
    average,
    cov,
    dtype,
    float64,
    full,
    ndarray,
    stack,
    uint8,
    unique,
    zeros,
)
from numpy.linalg import eigh as np_eigh
from numpy.random import permutation, seed
from scipy.linalg import eigh as scipy_eigh
from sklearn.datasets import load_iris

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type F64Array = ndarray[tuple[int], dtype[float64]]
type U8Array = ndarray[tuple[int], dtype[uint8]]

IRIS_LABELS: list[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_data() -> tuple[ndarray, ndarray]:
    load_res = cast(dict[str, ndarray], load_iris())
    data, target = load_res["data"], load_res["target"]
    data = data.T.copy()
    return data, target


def pca(data: F64Matrix, m: int):
    """Computes PCA projection matrix P from the data and the desired number of dimensions m to be kept"""
    n = data.shape[0]
    if n <= m:
        raise ValueError(f"Dimensions cannot be reduced from {n} to {m}!")
    empirical_cov = cov(data, bias=True)
    _, U = np_eigh(empirical_cov)
    P = U[:, ::-1][:, :m]
    return P


def get_bw_cov(data: F64Matrix, target: ndarray) -> tuple[F64Matrix, F64Matrix]:
    """Returns between and within class covariance matrices. The two summed equal the empyrical covariance for the dataset."""
    mu = data.mean(axis=1).reshape(data.shape[0], 1)
    unique_targets, samples_per_target = unique(target, return_counts=True)

    # between class covariance
    per_class_mean = [data[:, target == t].mean(axis=1) for t in unique_targets]
    per_class_mean = [mu_c.reshape(data.shape[0], 1) for mu_c in per_class_mean]
    per_class_sb = [(mu_c - mu) @ (mu_c - mu).T for mu_c in per_class_mean]
    per_class_sb = stack(per_class_sb)
    Sb = average(per_class_sb, axis=0, weights=samples_per_target)

    # within class covariance
    per_class_cov = stack(
        [cov(data[:, target == t], dtype=float64, bias=True) for t in unique_targets]
    )
    Sw = average(per_class_cov, axis=0, weights=samples_per_target)
    return Sb, Sw


def lda(data: F64Matrix, target: ndarray, m: int):
    """Returns LDA projection matrix W given m: the number of directions to be kept"""
    unique_targets = unique(target)
    assert m <= unique_targets.shape[0]
    Sb, Sw = get_bw_cov(data, target)
    _, U = scipy_eigh(Sb, Sw)
    W = U[:, ::-1][:, :m]
    return W


def split_train_test(
    data: F64Matrix, target: ndarray
) -> tuple[F64Matrix, ndarray, F64Matrix, ndarray]:
    train_count = int(data.shape[1] * 2 / 3)
    seed(0)
    idx = permutation(data.shape[1])
    train_idx, test_idx = idx[:train_count], idx[train_count:]
    train_data, train_target = data[:, train_idx], target[train_idx]
    test_data, test_target = data[:, test_idx], target[test_idx]
    return train_data, train_target, test_data, test_target


def get_euclidean_threshold(data: F64Array, target: U8Array) -> float64:
    """WARNING: works for binary classification with one dimensional data."""
    assert data.shape[0] == data.size
    assert len(data.shape) == 1
    unique_targets = unique(target)
    assert unique_targets.size == 2

    return stack([data[target == t].mean() for t in unique_targets]).mean()


def euclidean_error_rate(
    data: F64Array, threshold: float64, ground_truth: U8Array
) -> float64:
    """WARNING: works for binary classification with one dimensional data."""
    assert data.shape[0] == data.size
    assert len(data.shape) == 1
    unique_targets = unique(ground_truth)
    assert unique_targets.size == 2

    predictions = full(data.shape, min(unique_targets), dtype=uint8)
    predictions[data > threshold] = max(unique_targets)
    return (predictions != ground_truth).sum() / ground_truth.size


def pca_lda_comparison():
    data, target = load_data()
    # removing iris setosa samples
    data = data[:, target != 0]
    target = target[target != 0]
    train_data, train_target, test_data, test_target = split_train_test(data, target)

    # compute error rate for LDA only
    W = lda(train_data, train_target, m=1)
    projected_train_data = (W.T @ train_data).flatten()
    projected_test_data = (W.T @ test_data).flatten()
    threshold = get_euclidean_threshold(projected_train_data, train_target)
    err_rate = euclidean_error_rate(projected_test_data, threshold, test_target)
    print(f"LDA error rate: {err_rate:.4f}")

    # error rate for PCA only
    P = -pca(train_data, 1)  # sign matters
    projected_train_data = (P.T @ train_data).flatten()
    projected_test_data = (P.T @ test_data).flatten()
    threshold = get_euclidean_threshold(projected_train_data, train_target)
    err_rate = euclidean_error_rate(projected_test_data, threshold, test_target)
    print(f"PCA error rate: {err_rate:.4f}")

    # error rate for PCA + LDA
    P = -pca(train_data, 3)
    projected_train_data = P.T @ train_data
    W = lda(projected_train_data, train_target, 1)
    projected_train_data = (W.T @ projected_train_data).flatten()
    threshold = get_euclidean_threshold(projected_train_data, train_target)
    projected_test_data = (W.T @ P.T @ test_data).flatten()
    err_rate = euclidean_error_rate(projected_test_data, threshold, test_target)
    print(f"PCA3 + LDA error rate: {err_rate:.4f}")


def main():
    pca_lda_comparison()


if __name__ == "__main__":
    main()
