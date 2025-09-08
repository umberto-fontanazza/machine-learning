from pathlib import Path

from lib import load_from_csv, split_train_test
from lib.application import TEST_APPLICATIONS
from lib.binary import cost_from_fn_fp
from lib.evaluation import dcf, dcf_min_bin
from lib.model import PCA_LDA_euclid_binary
from lib.mvg import Mvg
from lib.normal import normal_density
from lib.types import F64Matrix, U8Array
from matplotlib.pyplot import hist, plot, show, title
from numpy import array, cov, diag, float64, linspace, uint8, unique
from sklearn.metrics import confusion_matrix

cls_color = ["red", "blue"]


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
    train_data, train_target, test_data, test_target = split_train_test(data, target)
    for model in [
        Mvg(train_data, train_target),
        Mvg(train_data, train_target, tied=True),
        Mvg(train_data, train_target, naive=True),
    ]:
        print(model.name)
        for app in TEST_APPLICATIONS[0:3]:
            eff_pi = app.effective_prior
            prior = array([1 - eff_pi, eff_pi], dtype=float64)
            cost = cost_from_fn_fp(1, 1)

            scores = model.score(test_data, prior)
            pred = model.inference(test_data, prior)
            cm = confusion_matrix(test_target, pred).T
            emp_risk = dcf(cm, cost, prior)
            dcf_min = dcf_min_bin(scores, test_target, cost, prior)

            print(f"π = {eff_pi}\t DCF: {emp_risk:.3f}\tDCFmin: {dcf_min:.3f}")
        print()


if __name__ == "__main__":
    main()
