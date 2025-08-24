from pathlib import Path

from lib import load_from_csv
from lib.model import PCA_LDA_euclid_binary
from lib.normal import normal_density
from lib.types import F64Matrix, U8Array
from matplotlib.pyplot import hist, plot, show, title
from numpy import array, cov, diag, exp, float64, linspace, uint8, unique
from numpy.random import permutation, seed

cls_color = ["red", "blue"]


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


def test_pca_lda_euclid():
    """Test accuracy of a model applying PCA, LDA and and euclidean classifier on the data with various hyperparameters."""
    data, target = load_from_csv(Path(Path(__file__).parent, "train-data.csv"))
    target = target.astype(uint8)
    train_data, train_target, test_data, test_target = split_train_test(data, target)

    for pca_m in reversed(range(2, 7)):
        model = PCA_LDA_euclid_binary(train_data, train_target, pca_m)
        error_rate = model.inference(test_data, test_target)
        error_rate_percentage = error_rate * 100
        print(f"With pca_m = {pca_m} error rate was {error_rate_percentage:05.2f} %")


def univariate_fit(data: F64Matrix, target: U8Array):
    n_features: int = data.shape[0]
    unique_targets = unique(target)

    cls_means: list[F64Matrix] = [
        data[:, target == t].mean(axis=1).reshape(n_features, 1) for t in unique_targets
    ]
    cls_variances: list[F64Matrix] = [
        diag(cov(data[:, target == t], bias=True, dtype=float64)).reshape(n_features, 1)
        for t in unique_targets
    ]

    for feature in range(n_features):
        for t in unique_targets:
            mean = cls_means[t][feature, :].reshape(1, 1)
            variance = cls_variances[t][feature, :].reshape(1, 1)
            x = linspace(-5, 5, 1000, dtype=float64).reshape(1, 1000)
            y = normal_density(x, mean, variance)
            hist(
                data[feature, target == t],
                bins=30,
                density=True,
                alpha=0.4,
                color=cls_color[t],
            )
            plot(x.flatten(), y, alpha=0.4, color=cls_color[t])
            title(f"Feature {feature}")
        show()


def main():
    data, target = load_from_csv(Path(Path(__file__).parent, "train-data.csv"))
    target = target.astype(uint8)
    univariate_fit(data, target)


if __name__ == "__main__":
    main()
