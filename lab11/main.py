from itertools import product

from lib.data import load_iris
from lib.evaluation import dcf, dcf_min_bin, uniform_cost
from lib.svm import Svm
from numpy import full
from sklearn.metrics import confusion_matrix
from solution import train_dual_SVM_linear


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)

    n_classes = 2
    cost = uniform_cost(n_classes)
    prior = full((n_classes), 1 / n_classes)

    for K, C in product([1, 10], [0.1, 1, 10]):
        model = Svm(train_data, train_target, C, K)
        primal, dual = model.primal_loss, model.dual_loss
        scores = model.score(test_data)
        predicted_test_target = scores > 0
        err_rate = (predicted_test_target != test_target).sum() / test_target.size
        err_rate_percent = 100 * err_rate
        cm = confusion_matrix(test_target, predicted_test_target)

        print(
            f"K: {K:2}\t\tC: {C:4}\t\tprimal: {primal:5.2f}\t\tdual: {dual:5.2f}\t\t"
            + f"err rate: {err_rate_percent:5.1f}%\t\t"
            + f"dcf min: {dcf_min_bin(scores, test_target, cost, prior):.4f}\t\t"
            + f"dcf: {dcf(cm, cost, prior):.4f}\t\t"
        )


if __name__ == "__main__":
    main()
