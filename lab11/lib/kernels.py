from typing import Callable

from lib.types import F64Matrix
from numpy import exp


def inner_product(xis: F64Matrix, xjs: F64Matrix) -> F64Matrix:
    return xis.T @ xjs


def quadratic_kernel(xis: F64Matrix, xjs: F64Matrix) -> F64Matrix:
    return (xis.T @ xjs + 1) ** 2


def make_poly_kernel(
    degree: int, bias: float
) -> Callable[[F64Matrix, F64Matrix], F64Matrix]:
    return lambda xis, xjs: (xis.T @ xjs + bias) ** degree


def make_radial_kernel(gamma: float) -> Callable[[F64Matrix, F64Matrix], F64Matrix]:
    def radial_kernel(xis, xjs):
        return exp(
            -gamma
            * (
                -2 * xis.T @ xjs
                + (xis**2).sum(axis=0).reshape(-1, 1)
                + (xjs**2).sum(axis=0).ravel()
            )
        )

    return radial_kernel
