from typing import Callable

from lib.data import to_signed_target
from lib.types import F64Array, F64Matrix, U8Array
from numpy import append, average, exp, float64, log, logaddexp, zeros
from scipy.optimize import fmin_l_bfgs_b


def make_logreg_obj(
    data: F64Matrix, target: U8Array, regularization_coeff: float
) -> Callable[[F64Array], tuple[float, F64Array]]:
    z = to_signed_target(target)

    def logreg_obj(model_params: F64Array) -> tuple[float, F64Array]:
        w, b = model_params[0:-1], model_params[-1]
        scores = w @ data + b
        zs = z * scores
        neg_zs = -zs

        w_col = w.reshape((-1, 1))
        norm_w_squared = w_col.T @ w_col
        regularizer = regularization_coeff * norm_w_squared / 2
        logistic_loss = logaddexp(0, neg_zs)
        f_val = float(regularizer + average(logistic_loss))

        G = -z / (1 + exp(zs))
        grad_w = regularization_coeff * w + average(G * data, axis=1)
        deriv_b = average(G)

        return f_val, append(grad_w, [deriv_b])

    return logreg_obj


class LogregBin:
    w: F64Array
    b: float64
    obj_val: float
    prior: float

    def __init__(self, data: F64Matrix, target: U8Array, reg_coeff: float):
        obj_fn = make_logreg_obj(data, target, reg_coeff)
        optim, obj_val, _ = fmin_l_bfgs_b(obj_fn, zeros((data.shape[0] + 1)))
        self.w, self.b = optim[:-1], optim[-1]
        self.obj_val = obj_val
        self.prior = (target > 0).sum() / target.size

    def scores(self, data: F64Matrix) -> F64Array:
        """Scores reflect training set prior"""
        return self.w @ data + self.b

    def llr(self, data: F64Matrix) -> F64Array:
        return self.scores(data) - log(self.prior / (1 - self.prior))
