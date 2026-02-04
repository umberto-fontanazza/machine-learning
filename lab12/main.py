from itertools import product
from pathlib import Path

from lib.data import load_iris
from lib.gmm import CovarianceType, logpdf_gmm, train_gmm
from numpy import argmax, unique, vstack
from solution.gmm import save_gmm

DATA_PATH = Path(__file__).parent / "data"


def main():
    train_data, train_target, test_data, test_target = load_iris()

    min_eig = 0.01
    stop_delta = 1e-6
    alpha = 0.1

    for n_comp in [1, 2, 4, 8, 16]:
        print(f"Components: {n_comp}")
        for cov_type in [CovarianceType.FULL, CovarianceType.DIAG, CovarianceType.TIED]:

            cls_gmms = []
            for trg in unique(train_target):
                data = train_data[:, train_target == trg]
                gmm = train_gmm(
                    data,
                    alpha,
                    stop_delta,
                    n_comp,
                    cov_type,
                    min_eig,
                )
                cls_gmms.append(gmm)
            class_cond_logpdf = vstack([logpdf_gmm(test_data, gmm) for gmm in cls_gmms])
            predicted = argmax(class_cond_logpdf, axis=0)
            err_rate = float((predicted != test_target).sum() / test_target.size * 100)
            print(f"{cov_type.name}: {err_rate} %")
        print()


if __name__ == "__main__":
    main()
