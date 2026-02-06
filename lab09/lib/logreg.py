from typing import Callable, cast

from lib.data import extend_with_bias, to_signed_target
from lib.types import F64Array, F64Matrix, U8Array
from numpy import append, average, exp, float64, logaddexp, zeros
from scipy.optimize import fmin_l_bfgs_b


def make_logreg_obj(
    data: F64Matrix, target: U8Array, regularization_coeff: float
) -> Callable[[F64Array], tuple[float, F64Array]]:
    z = to_signed_target(target)

    assert data.shape[0] == 4
    assert data.shape[1] == target.size
    assert target.ndim == 1

    def logreg_obj(model_params: F64Array) -> tuple[float, F64Array]:
        assert model_params.size == data.shape[0] + 1
        w, b = model_params[0:-1], model_params[-1]
        assert w.size == data.shape[0]
        scores = w @ data + b
        assert scores.size == data.shape[1]
        assert scores.ndim == 1
        neg_zs = -z * scores

        w_col = w.reshape((-1, 1))
        norm_w_squared = w_col.T @ w_col
        regularizer = regularization_coeff * norm_w_squared / 2
        logistic_loss = logaddexp(0, neg_zs)
        f_val = float(regularizer + average(logistic_loss))

        G = -z / (1 + exp(neg_zs))
        grad_w = regularization_coeff * w + average(G * data, axis=1)
        deriv_b = average(G)

        return f_val, append(grad_w, [deriv_b])

    return logreg_obj


class Logreg:
    w: F64Array
    b: float64
    obj_val: float

    def __init__(self, data: F64Matrix, target: U8Array, reg_coeff: float):
        obj_fn = make_logreg_obj(data, target, reg_coeff)
        optim, obj_val, _ = fmin_l_bfgs_b(obj_fn, zeros((data.shape[0] + 1)))
        self.w, self.b = optim[:-1], optim[-1]
        self.obj_val = obj_val

    def score(self, data: F64Matrix) -> F64Array:
        return self.w @ data + self.b
