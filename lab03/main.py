from typing import cast

from matplotlib.pyplot import scatter, show
from numpy import array, average, cov, dtype, float64, ndarray, stack, unique
from numpy.linalg import eigh as np_eigh
from numpy.random import permutation, seed
from scipy.linalg import eigh as scipy_eigh
from sklearn.datasets import load_iris

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]


def load_data() -> tuple[ndarray, ndarray]:
    load_res = cast(dict[str, ndarray], load_iris())
    data, target = load_res["data"], load_res["target"]
    data = data.T.copy()
    return data, target


def pca(data, m: int):
    """Apply dimensionality reduction to the provided dataset reducing from data.shape[0] dimensions to m dimensions."""
    n = data.shape[0]
    if n <= m:
        raise ValueError(f"Dimensions cannot be reduced from {n} to {m}!")
    empirical_cov = cov(data, bias=True)
    _, U = np_eigh(empirical_cov)
    P = U[:, ::-1][:, :m]
    return P.T @ data  # projected_data


def pca_example(data, target):
    projected_data = pca(data, 2)
    assert projected_data.shape[0] == 2
    assert projected_data.shape[1] == 150
    for cls in range(3):
        scatter(projected_data[0, target == cls], projected_data[1, target == cls])
    show()


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
    assert m <= max(target)
    Sb, Sw = get_bw_cov(data, target)
    _, U = scipy_eigh(Sb, Sw)
    m = 2
    W = U[:, ::-1][:, :m]
    # Check against professor's solution
    # W_solution = load(Path(Path(__file__).parent, "solution", "IRIS_LDA_matrix_m2.npy"))
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


def main():
    data, target = load_data()
    train_data, train_target, test_data, test_target = split_train_test(data, target)


if __name__ == "__main__":
    main()
