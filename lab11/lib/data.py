from typing import cast

from numpy import full, int8, ndarray, unique, vstack
from numpy.random import permutation
from numpy.random import seed as np_seed
from sklearn.datasets import load_iris as sklearn_load_iris

from .types import F64Matrix, I8Array, U8Array

SPLIT_SEED = 0


def load_iris(binary=False) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    """Returns train_data, train_target, test_data, test_target"""
    load_res = cast(dict[str, ndarray], sklearn_load_iris())
    data, target = load_res["data"].T, load_res["target"]
    if binary:
        data = data[:, target != 0]
        target = target[target != 0]
    target[target == 2] = 0  # WARNING: this may be a source of errors
    return split_train_test(data, target)


def split_train_test(
    data: F64Matrix, target: U8Array, seed=SPLIT_SEED, train_fraction=2 / 3
) -> tuple[F64Matrix, U8Array, F64Matrix, U8Array]:
    """Returns train_data, train_target, test_data, test_target"""
    tot_samples = data.shape[1]
    train_count = int(tot_samples * train_fraction)
    np_seed(seed)
    idx = permutation(data.shape[1])
    train_idx, test_idx = idx[:train_count], idx[train_count:]
    return data[:, train_idx], target[train_idx], data[:, test_idx], target[test_idx]


def extend_with_bias(data: F64Matrix, bias: float) -> F64Matrix:
    """Extends data matrix by concatenating bias to each sample"""
    return vstack((data, full((1, data.shape[1]), bias)))


def to_signed_target(target: ndarray) -> I8Array:
    """Takes an ndarray of labels and turns it into [-1, 1] format."""
    unique_labels = unique(target)
    assert unique_labels.size == 2
    fls = min(unique_labels)
    transformed_target = full((target.size), 1, dtype=int8)
    transformed_target[target == fls] = -1
    return transformed_target
