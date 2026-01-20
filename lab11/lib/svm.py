from functools import cache
from typing import Callable, cast

from lib.data import to_signed_target
from lib.kernels import inner_product
from lib.types import F64Array, F64Matrix, I8Array, Kernel, U8Array
from numpy import float64, maximum
from numpy import nan as NAN
from numpy import sum, zeros
from scipy.optimize import fmin_l_bfgs_b


def _primal_loss(w_hat: F64Array, x_hat: F64Matrix, z: I8Array, C: float) -> float64:
    """
    ### Parameters
    :param w_hat: concatenation of w, b defining separation hyperplane
    :param x_hat: train set samples
    :param z: train set labels (bipolar form: +1 = True, -1 = False)
    :param C: SVM hyperparameter
    """
    _w_hat = w_hat.reshape((-1, 1))

    return (
        0.5 * (_w_hat.T @ _w_hat)
        + C
        * maximum(
            0,
            (1 - (_w_hat.T @ x_hat) * z),
        ).sum()
    ).item()


def _make_dual_loss_deriv(
    H_hat: F64Matrix,
) -> Callable[[F64Array], tuple[float64, F64Array]]:
    def L_dual(
        alpha: F64Array,
    ) -> tuple[float64, F64Array]:
        a = alpha.reshape((-1, 1))
        Ha = H_hat @ a
        minus_J = cast(
            float64,
            (0.5 * a.T @ Ha).item() - sum(a),
        )
        minus_grad_J = cast(F64Array, (Ha - 1).ravel())
        return minus_J, minus_grad_J

    return L_dual


def train_svm(
    xis: F64Matrix, z: I8Array, C: float, kernel_fn: Kernel
) -> tuple[F64Array, float64, F64Matrix]:
    """
    ### Parameters
    :param x_hat: train data extended with bias
    :param z: train target (labels) in bipolar (+1=True, -1=False) form

    ### Return value
    Tuple with 3 fields
    1 - alpha_otim: alphas solution of the dual objective
    2 - dual_loss: value of the dual loss at minimum
    3 - h_hat: z @ z.T * kern(x_hat, x_hat)
    """
    n_samples = xis.shape[1]
    _z = z.reshape((-1, 1))

    H_hat = _z @ _z.T * kernel_fn(xis, xis)

    alpha_optim, minus_dual_loss, _ = fmin_l_bfgs_b(
        _make_dual_loss_deriv(H_hat),
        zeros(n_samples),
        bounds=[(0, C) for _ in range(n_samples)],
        factr=NAN,
        pgtol=1e-5,
    )

    return alpha_optim, -minus_dual_loss, H_hat


class Svm:
    xis: F64Matrix
    z: I8Array
    C: float
    kernel_fn_hat: Kernel
    alpha_optim: F64Array
    H_hat: F64Matrix
    dual_loss: float

    def __init__(
        self,
        train_data: F64Matrix,
        train_target: U8Array,
        C: float,
        K: float = 1,
        kernel_fn: Kernel = inner_product,
    ):
        z = to_signed_target(train_target)
        kernel_fn_hat = lambda xis, xjs: kernel_fn(xis, xjs) + K**2
        self.xis = train_data
        self.z = z
        self.C = C
        self.kernel_fn_hat = kernel_fn_hat

        self.alpha_optim, dual_loss64, self.H_hat = train_svm(
            train_data, z, C, kernel_fn_hat
        )
        self.dual_loss = float(dual_loss64)

    def score(self, test_data: F64Matrix) -> F64Array:
        coeffs = (self.alpha_optim * self.z).reshape((-1, 1))
        res = (coeffs * self.kernel_fn_hat(self.xis, test_data)).sum(axis=0).ravel()
        return res

    @property
    @cache
    def primal_loss(self) -> float:
        _alpha_optim, C, H_hat = self.alpha_optim.reshape((-1, 1)), self.C, self.H_hat
        Ha = H_hat @ _alpha_optim
        return float((_alpha_optim.T @ Ha / 2 + C * maximum(0, 1 - Ha).sum()).item())
