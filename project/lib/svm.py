from functools import cache

from lib.data import to_bipolar_target
from lib.kernel import inner_product
from lib.types import F64Array, F64Matrix, I8Array, Kernel, U8Array
from numpy import maximum
from numpy import nan as NaN
from numpy import zeros
from scipy.optimize import fmin_l_bfgs_b


class Svm:
    train_data: F64Matrix
    train_target: I8Array  # bipolar form [-1, 1]
    C: float
    K: float
    alpha_optim: F64Array
    dual_loss: float
    kern_fn: Kernel

    def __init__(
        self,
        train_data: F64Matrix,
        train_target: U8Array,
        C: float,
        K: float,
        kern_fn: Kernel = inner_product,
    ):
        z = to_bipolar_target(train_target)
        _z = z.reshape((-1, 1))
        H_hat = _z @ _z.T * kern_fn(train_data, train_data) + K**2

        def dual_loss(alpha: F64Array) -> tuple[float, F64Array]:
            Ha = H_hat @ alpha.reshape((-1, 1))
            loss = alpha @ Ha / 2 - alpha.sum()
            grad = (Ha - 1).ravel()
            return loss, grad

        n_samples = train_data.shape[1]

        alpha_star, minus_dual_loss, _ = fmin_l_bfgs_b(
            dual_loss,
            zeros(n_samples),
            bounds=[(0, C) for _ in range(n_samples)],
            factr=NaN,
            pgtol=1e-5,
        )

        self.C = C
        self.K = K
        self.kern_fn = kern_fn
        self.train_data = train_data
        self.train_target = z
        self.alpha_optim = alpha_star
        self.dual_loss = float(-minus_dual_loss)

    def score(self, data: F64Matrix) -> F64Array:
        a = self.alpha_optim.reshape((-1, 1))
        z = self.train_target.reshape((-1, 1))
        K = self.kern_fn(self.train_data, data) + self.K**2
        return (a * z * K).sum(axis=0)

    @property
    @cache
    def primal_loss(self) -> float:
        alpha = self.alpha_optim.reshape((-1, 1))
        z = self.train_target.reshape((-1, 1))

        norm_w_hat: float = (
            alpha
            @ alpha.T
            * (z @ z.T)
            * (self.kern_fn(self.train_data, self.train_data) + self.K**2)
        ).sum()

        total_hinge_loss: float = maximum(
            0, 1 - z.T * self.score(self.train_data)
        ).sum()

        return norm_w_hat / 2 + self.C * total_hinge_loss
