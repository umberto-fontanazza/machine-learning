from pathlib import Path
from typing import cast

from lib.mvg import Mvg
from lib.types import F64Array, F64Matrix, U8Array
from numpy import (
    array,
    astype,
    dtype,
    exp,
    float64,
    int64,
    load,
    log,
    ndarray,
    ones,
    uint8,
)
from numpy.random import permutation, seed
from scipy.special import logsumexp
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


def load_data() -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    data, target = load_iris(return_X_y=True)
    data = data.T
    target = astype(cast(ndarray[tuple[int], dtype[int64]], target), uint8)
    return split_train_test(data, target)


def split_train_test(
    data: F64Matrix, target: U8Array
) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    """Splits into train and test set with 2/3 of samples assigned to train and 1/3 to test.
    Returns train_data, train_target, test_data, test_target."""
    seed(0)
    idx = permutation(data.shape[1])
    train_count = int(data.shape[1] * 2 / 3)
    train_idx, test_idx = idx[:train_count], idx[train_count:]
    train_data, train_target = data[:, train_idx], target[train_idx]
    test_data, test_target = data[:, test_idx], target[test_idx]
    return train_data, train_target, test_data, test_target


def mvg_confusion():
    train_data, train_target, test_data, test_target = load_data()
    model = Mvg(train_data, train_target)
    print(model.confusion(test_data, test_target))
    model_tied = Mvg(train_data, train_target, tied=True)
    print(model_tied.confusion(test_data, test_target))


def commedia_confusion():
    cls_cond_ll = load(Path("data", "commedia_ll.npy"))
    target = load(Path("data", "commedia_labels.npy")).astype(uint8)
    prior = normalize(ones((3, 1), dtype=float64), norm="l1", axis=0)
    log_prior = log(prior)
    log_joint: F64Matrix = cls_cond_ll + log_prior
    predicted: U8Array = log_joint.argmax(axis=0).astype(uint8)
    print(confusion_matrix(target, predicted).T)


def cost_aware_classify(
    llr: F64Array, pi_true: float, cost_fn: float, cost_fp: float
) -> U8Array:
    log_joint = llr + log(pi_true / (1 - pi_true))
    log_cost = log(cost_fn / cost_fp)
    return ((log_joint + log_cost) > 0).astype(uint8)


def binary_cost_matrix(cost_fn: float, cost_fp: float) -> F64Matrix:
    return array([[0, cost_fn], [cost_fp, 0]])


def bayes_risk(confusion_matrix: F64Matrix, cost: F64Matrix, prior: F64Array) -> float:
    assert confusion_matrix.ndim == 2
    assert cost.ndim == 2
    assert prior.ndim == 1
    return (
        (cost * confusion_matrix).sum(axis=0) / confusion_matrix.sum(axis=0) * prior
    ).sum()


def dcf(conf_m: F64Matrix, cost: F64Matrix, prior: F64Array) -> float:
    return bayes_risk(conf_m, cost, prior)


def min_dcf(
    llr: F64Array, target: U8Array, pi: float, cost_fn: float, cost_fp: float
) -> float:
    sorted_target = target[llr.argsort()]

    predicted_labels = ones((llr.size), dtype=uint8)
    conf_m = confusion_matrix(target, predicted_labels).T
    cost = binary_cost_matrix(cost_fn, cost_fp)
    prior = array([(1 - pi), pi])
    min_dcf = dcf(conf_m, cost, prior)
    for target in sorted_target:
        conf_m[1, target] -= 1
        conf_m[0, target] += 1
        min_dcf = min(min_dcf, dcf(conf_m, cost, prior))
    return min_dcf


def main():
    cls_conditional_llr = load("data/commedia_llr_infpar.npy")
    target = load("data/commedia_labels_infpar.npy")
    applications = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]
    for pi, cost_fn, cost_fp in applications:
        predictions = cost_aware_classify(cls_conditional_llr, pi, cost_fn, cost_fp)
        m = confusion_matrix(target, predictions).T
        risk = bayes_risk(m, binary_cost_matrix(cost_fn, cost_fp), array([1 - pi, pi]))
        base_risk = min(pi * cost_fn, (1 - pi) * cost_fp)

        normalized_dcf = risk / base_risk
        print(f"{normalized_dcf=}")

        minimum_dcf = min_dcf(cls_conditional_llr, target, pi, cost_fn, cost_fp)
        min_norm_dcf = minimum_dcf / base_risk
        print(f"{min_norm_dcf=}")


if __name__ == "__main__":
    main()
