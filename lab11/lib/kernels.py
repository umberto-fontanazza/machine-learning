from typing import Callable

from lib.types import F64Matrix


def inner_product(xis: F64Matrix, xjs: F64Matrix) -> F64Matrix:
    return xis.T @ xjs


def quadratic_kernel(xis: F64Matrix, xjs: F64Matrix) -> F64Matrix:
    return (xis.T @ xjs + 1) ** 2


def make_poly_kernel(
    degree: int, bias: float
) -> Callable[[F64Matrix, F64Matrix], F64Matrix]:
    return lambda xis, xjs: (xis.T @ xjs + bias) ** degree
