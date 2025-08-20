from pathlib import Path

from lib import load_from_csv
from lib.model import PCA_LDA_euclid_binary
from lib.types import F64Matrix, U8Array
from numpy import uint8
from numpy.random import permutation, seed


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


def main():
    data, target = load_from_csv(Path(Path(__file__).parent, "train-data.csv"))
    target = target.astype(uint8)
    train_data, train_target, test_data, test_target = split_train_test(data, target)

    for pca_m in reversed(range(2, 7)):
        model = PCA_LDA_euclid_binary(train_data, train_target, pca_m)
        error_rate = model.inference(test_data, test_target)
        error_rate_percentage = error_rate * 100
        print(f"With pca_m = {pca_m} error rate was {error_rate_percentage:05.2f} %")


if __name__ == "__main__":
    main()
