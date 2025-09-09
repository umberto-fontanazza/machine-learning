from typing import Callable, cast

from numpy import append, dtype, exp, float64, log, logaddexp, ndarray, uint8, zeros
from numpy.random import permutation, seed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.datasets import load_iris

type F64Array = ndarray[tuple[int, int], dtype[float64]]
type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type U8Array = ndarray[tuple[int], dtype[uint8]]


def split_train_test(
    data: F64Matrix, target: U8Array
) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    tot_samples = data.shape[1]
    train_count = int(tot_samples * 2 / 3)
    seed(0)
    idx = permutation(data.shape[1])
    train_idx, test_idx = idx[:train_count], idx[train_count:]
    return data[:, train_idx], target[train_idx], data[:, test_idx], target[test_idx]


def load_iris_binary() -> tuple[F64Matrix, U8Array]:
    load_res = cast(dict[str, ndarray], load_iris())
    data, target = load_res["data"].T, load_res["target"]
    data = data[:, target != 0]  # We remove setosa from D
    target = target[target != 0]  # We remove setosa from L
    target[target == 2] = 0  # We assign label 0 to virginica (was label 2) return D, L
    return data, target


def curry_logreg_obj(
    data: F64Matrix, target: U8Array, lam: float
) -> Callable[[F64Array], tuple[float64, F64Array]]:
    n = data.shape[1]

    def logreg_obj(w_b: F64Array) -> tuple[float64, F64Array]:
        assert w_b.ndim == 1

        w, b = w_b[:-1].reshape(1, -1), w_b[-1]

        scores = w @ data + b
        z = 2 * target - 1

        logreg_obj_val = (
            0.5 * lam * (w**2).sum()  # regularization term
            + 1 / n * logaddexp(0, (-z) * scores).sum()
        )

        g_term = (-z) / (1 + exp(z * scores))
        grad_w = lam * w + 1 / n * (g_term * data).sum(axis=1)
        grad_b = 1 / n * g_term.sum()
        grad = append(grad_w, grad_b)
        return logreg_obj_val, grad

    return logreg_obj


def main():
    data, target = load_iris_binary()
    train_data, train_target, test_data, test_target = split_train_test(data, target)
    logreg_obj = curry_logreg_obj(train_data, train_target, 1)

    x, y, metadata = fmin_l_bfgs_b(logreg_obj, zeros(train_data.shape[0] + 1))
    print(x, y)

    w, b = x[:-1], x[-1]
    pi = 0.5
    log_odds = log(pi / (1 - pi))
    scores = w @ test_data + b - log_odds
    predicted = (scores > 0).astype(uint8)
    e_rate = (predicted != test_target).sum() / predicted.size
    print(f"{e_rate=}")


if __name__ == "__main__":
    main()
