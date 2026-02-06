from lib.data import load_iris
from lib.logreg import Logreg
from numpy import uint8


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)
    print(f"{train_data.shape=}")
    for reg_coeff in [1e-3, 1e-1, 1]:
        model = Logreg(train_data, train_target, reg_coeff)
        scores = model.score(test_data)
        predicted = (scores > 0).astype(uint8)
        err_rate = (predicted != test_target).sum() / test_target.size * 100
        print(f"lambda = {reg_coeff:.5f} | {model.obj_val} | {err_rate}")


if __name__ == "__main__":
    main()
