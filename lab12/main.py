from pathlib import Path

from lib.data import load_iris, split_train_test
from lib.evaluation import dcf, dcf_min_bin
from lib.gmm import CovarianceType, GmmClassifier
from numpy import array, exp, load, uint8
from sklearn.metrics import confusion_matrix

DATA_PATH = Path(__file__).parent / "data"


def main():
    data = load(DATA_PATH / "ext_data_binary.npy")
    target = load(DATA_PATH / "ext_data_binary_labels.npy")
    train_data, train_target, test_data, test_target = split_train_test(data, target)

    min_eig = 0.01
    stop_delta = 1e-6
    alpha = 0.1

    for target_components in [1, 2, 4, 8, 16]:
        print(f"Components: {target_components}")
        for cov_type in [CovarianceType.FULL, CovarianceType.DIAG, CovarianceType.TIED]:

            model = GmmClassifier(
                train_data,
                train_target,
                alpha,
                stop_delta,
                target_components,
                cov_type,
                min_eig,
            )
            class_cond_logpdf = model.class_cond_logpdf(test_data)
            llr = (class_cond_logpdf[1, :] - class_cond_logpdf[0, :]).ravel()
            cost = array([[0, 1], [1, 0]])
            prior = array([0.5, 0.5])
            pred = (llr > 0).astype(uint8)
            conf_m = confusion_matrix(pred, test_target).T

            _dcf = dcf(conf_m, cost, prior)
            dcf_min = dcf_min_bin(llr, test_target, cost, prior)
            print(f"{dcf_min:.4f} / {_dcf:.4f}")
        print()


if __name__ == "__main__":
    main()
