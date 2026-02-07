from lib.data import load_iris
from lib.evaluation import dcf, dcf_min_bin
from lib.logreg import LogregBin
from numpy import array, uint8
from sklearn.metrics import confusion_matrix


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)
    for reg_coeff in [1e-3, 1e-1, 1]:
        model = LogregBin(train_data, train_target, reg_coeff)
        llr = model.llr(test_data)
        predicted = (llr > 0).astype(uint8)

        conf_m = confusion_matrix(test_target, predicted).T
        cost = array([[0, 1], [1, 0]])
        prior = array([0.5, 0.5])
        dcf_act = dcf(conf_m, cost, prior)
        dcf_min = dcf_min_bin(llr, test_target, cost, prior)
        err_rate = (predicted != test_target).sum() / test_target.size * 100

        print(
            f"lambda = {reg_coeff:.5f} | {model.obj_val:.5f} | {err_rate:.3f} | {dcf_min:.4f} | {dcf_act:.4f}"
        )


if __name__ == "__main__":
    main()
