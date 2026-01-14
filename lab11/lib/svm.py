from typing import Callable, cast

from lib.data import extend_with_bias, to_signed_target
from lib.types import F64Array, F64Matrix, I8Array, U8Array
from numpy import concatenate, float64, maximum
from numpy import nan as NAN
from numpy import sum, zeros
from scipy.optimize import fmin_l_bfgs_b


def _svm_primal_loss(
    w_hat: F64Array, x_hat: F64Matrix, z: I8Array, C: float
) -> float64:
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
    x_hat: F64Matrix,
    z: I8Array,
    C: float,
    K: float,
) -> tuple[F64Array, float64, float64]:
    """
    ### Parameters
    :param x_hat: train data extended with bias
    :param z: train target (labels) in bipolar (+1=True, -1=False) form

    ### Return value
    Tuple with 3 fields
    1 - w: hyperplane normal vec
    2 - b: bias
    3 - dual_loss: value of the dual loss at minimum
    """
    n_samples = x_hat.shape[1]
    _z = z.reshape((-1, 1))

    H_hat = _z @ _z.T * (x_hat.T @ x_hat)

    alpha_optim, minus_dual_loss, _ = fmin_l_bfgs_b(
        _make_dual_loss_deriv(H_hat),
        zeros(n_samples),
        bounds=[(0, C) for _ in range(n_samples)],
        factr=NAN,
        pgtol=1e-5,
    )

    w_hat = (alpha_optim * _z.T * x_hat).sum(axis=1)
    w, b = w_hat[0:-1], w_hat[-1] * K
    return w, b, -minus_dual_loss


class Svm:
    C: float
    K: float
    w: F64Array
    b: float64
    primal_loss: float
    dual_loss: float

    def __init__(
        self,
        train_data: F64Matrix,
        train_target: U8Array,
        C: float,
        K: float = 1,
    ):
        x_hat, z = extend_with_bias(train_data, K), to_signed_target(train_target)
        self.C, self.K = C, K
        self.w, self.b, dual_loss64 = train_svm(
            x_hat,
            z,
            C,
            K,
        )
        self.dual_loss = float(dual_loss64)
        self.primal_loss = float(_svm_primal_loss(self.w_hat, x_hat, z, C))

    def score(self, test_data: F64Matrix) -> F64Array:
        return (self.w @ test_data + self.b).ravel()

    @property
    def w_hat(self) -> F64Array:
        return concatenate((self.w, [self.b / self.K]))
