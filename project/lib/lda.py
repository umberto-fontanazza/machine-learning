from numpy import average, cov, stack, unique
from scipy.linalg import eigh

from .types import F64Matrix, U8Array


def get_lda_lt(data: F64Matrix, target: U8Array, m: int) -> F64Matrix:
    """Returns the linear transformation of LDA to project the dataset into m dimensions. M must be (strictly) lower than the number of classes."""
    assert len(data.shape) == 2
    unique_targets, count_per_target = unique(target, return_counts=True)
    assert m < unique_targets.size
    mu = data.mean(axis=1).reshape(data.shape[0], 1)

    # compute Sb
    cls_means = [data[:, target == t].mean(axis=1) for t in unique_targets]
    cls_means = [mu_c.reshape(data.shape[0], 1) for mu_c in cls_means]
    Sb = average(
        stack([(mu_c - mu) @ (mu_c - mu).T for mu_c in cls_means]),
        axis=0,
        weights=count_per_target,
    )

    # compute Sw
    per_class_emp_cov = stack(
        [cov(data[:, target == t], bias=True) for t in unique_targets]
    )
    Sw = average(per_class_emp_cov, axis=0, weights=count_per_target)

    _, U = eigh(Sb, Sw)
    return U[:, ::-1][:, :m]
