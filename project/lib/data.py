from lib.types import F64Matrix, I8Array, U8Array
from numpy import full, unique, vstack


def extend_with_constant(data: F64Matrix, constant: float) -> F64Matrix:
    return vstack((data, full(data.shape[1], constant)))


def to_bipolar_target(target: U8Array) -> I8Array:
    unique_labels = unique(target)
    if unique_labels.size != 2:
        raise ValueError(
            "Target must have exactly two unique labels, representing a binary problem"
        )
    _, true_label = unique_labels
    bipolar = full(target.shape, -1)
    bipolar[target == true_label] = 1
    return bipolar
