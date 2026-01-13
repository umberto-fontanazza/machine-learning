from typing import cast

from lib.data import load_iris, to_signed_target
from lib.types import F64Array, F64Matrix, U8Array
from numpy import float64, full, sum, vstack, zeros
from scipy.optimize import fmin_l_bfgs_b
from solution import train_dual_SVM_linear


def svm(
    train_data: F64Matrix, train_target: U8Array, C: float, K=1
) -> tuple[F64Array, float64]:
    train_data_ext = vstack((train_data, full((1, train_data.shape[1]), K)))
    z = to_signed_target(train_target).reshape((-1, 1))
    G_hat = train_data_ext.T @ train_data_ext
    H_hat = z @ z.T * G_hat

    def L_dual(alpha: F64Array) -> tuple[float64, F64Array]:
        a = alpha.reshape((-1, 1))
        Ha = H_hat @ a
        minus_J = cast(float64, (0.5 * a.T @ Ha).item() - sum(a))
        minus_grad_J = cast(F64Array, (Ha - 1).ravel())
        return minus_J, minus_grad_J

    alpha_optim, _, _ = fmin_l_bfgs_b(
        L_dual,
        zeros((train_data_ext.shape[1]), dtype=float64),
        bounds=[(0, 0.1) for _ in range(train_data_ext.shape[1])],
    )

    w_hat = (alpha_optim * z.T * train_data_ext).sum(axis=1)
    w, b = w_hat[0:-1], w_hat[-1] * K
    return w, b


def main():
    train_data, train_target, test_data, test_target = load_iris(binary=True)
    print("MINE: ===========================")
    svm(train_data, train_target, 0.1)
    print("PROF: ===========================")
    train_dual_SVM_linear(train_data, train_target, 0.1)


if __name__ == "__main__":
    main()
