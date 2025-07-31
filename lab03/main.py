from functools import cache
from typing import cast

from matplotlib.pyplot import figure, scatter, show
from numpy import cov, ndarray
from numpy.linalg import eigh
from sklearn.datasets import load_iris


@cache
def load() -> tuple[ndarray, ndarray]:
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
    _, U = eigh(empirical_cov)
    P = U[:, ::-1][:, :m]
    return P.T @ data  # projected_data


def main():
    data, target = load()
    projected_data = pca(data, 2)
    assert projected_data.shape[0] == 2
    assert projected_data.shape[1] == 150
    for cls in range(3):
        scatter(projected_data[0, target == cls], projected_data[1, target == cls])
    show()


if __name__ == "__main__":
    main()
