from lib.types import F64Matrix, Kernel


def inner_product(xis: F64Matrix, xjs: F64Matrix) -> F64Matrix:
    return xis.T @ xjs
