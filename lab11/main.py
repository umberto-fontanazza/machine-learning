from itertools import product

from lib.data import load_iris
from lib.svm import Svm
from solution import train_dual_SVM_linear


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)

    for K, C in product([1, 10], [0.1, 1, 10]):
        model = Svm(train_data, train_target, C, K)
        primal, dual = model.primal_loss, model.dual_loss
        print(f"K: {K:2}\t\tC: {C:4}\t\tprimal: {primal:5.2f}\t\tdual: {dual:5.2f}")


if __name__ == "__main__":
    main()
