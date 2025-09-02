from typing import cast

from lib.mvg import Mvg
from lib.types import F64Matrix, U8Array
from numpy import astype, dtype, int64, ndarray, uint8
from numpy.random import permutation, seed
from sklearn.datasets import load_iris


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


def main():
    pass


if __name__ == "__main__":
    main()
