from csv import reader
from pathlib import Path

from lib.types import F64Matrix, U8Array
from numpy import array, float64, uint8
from numpy.random import permutation, seed


def load_from_csv(path: Path) -> tuple[F64Matrix, U8Array]:
    buffer: list[float] = []
    labels: list[bool] = []
    n_features: int = 0
    with open(path, "r") as csv_file:
        csv_reader = reader(csv_file)
        for index, line in enumerate(csv_reader):
            if index == 0:
                n_features = len(line) - 1
            buffer.extend([float(x) for x in line[:-1]])
            labels.append(int(line[-1]) != 0)
    n_samples: int = len(buffer) // n_features
    data: F64Matrix = array(buffer, dtype=float64)
    data = data.reshape(n_samples, n_features).T.copy()
    target: U8Array = array(labels, dtype=uint8)
    return data, target


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
