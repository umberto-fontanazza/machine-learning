from typing import cast

from lib.data import load_iris, to_signed_target
from lib.types import F64Array, F64Matrix, U8Array
from numpy import float64, ones, sum, vstack, zeros
from scipy.optimize import fmin_l_bfgs_b
from solution import train_dual_SVM_linear


def svm(train_data: F64Matrix, train_target: U8Array, C: float, K=1):
    train_data = vstack((train_data, ones((1, train_data.shape[1]))))
    z = to_signed_target(train_target).reshape((-1, 1))
    G = train_data.T @ train_data
    H = z @ z.T * G

    def L_dual(alpha: F64Array) -> tuple[float64, F64Array]:
        a = alpha.reshape((-1, 1))
        Ha = H @ a
        minus_J = cast(float64, (0.5 * a.T @ Ha).item() - sum(a))
        minus_grad_J = cast(F64Array, (Ha - 1).ravel())
        return minus_J, minus_grad_J

    alpha_optim, _, _ = fmin_l_bfgs_b(
        L_dual,
        zeros((train_data.shape[1]), dtype=float64),
        bounds=[(0, 0.1) for _ in range(train_data.shape[1])],
    )
    print(alpha_optim)

    w = (alpha_optim * z.T * train_data).sum(axis=1)  # this is w hat


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)
    print("MINE: ===========================")
    svm(train_data, train_target, 0.1)
    print("PROF: ===========================")
    train_dual_SVM_linear(train_data, train_target, 0.1)


if __name__ == "__main__":
    main()
