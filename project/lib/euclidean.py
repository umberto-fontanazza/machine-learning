from numpy import float64, full, stack, uint8, unique

from .types import F64Array, U8Array


def euclid_threshold(data: F64Array, target: U8Array) -> float64:
    """Returns the threshold for an euclidean classifier. The threshold is the average of the class means."""
    unique_targets = unique(target)
    assert unique_targets.size == 2
    return stack([data[target == t].mean() for t in unique_targets]).mean()


def euclid_classify(
    data: F64Array, threshold: float64, unique_targets: U8Array
) -> U8Array:
    predicted = full(data.shape, unique_targets[0], dtype=uint8)
    predicted[data > threshold] = unique_targets[1]
    return predicted


def euclid_error_rate(
    data: F64Array, threshold: float64, gorund_truth: U8Array
) -> float:
    unique_targets = unique(gorund_truth)
    assert unique_targets.size == 2
    predicted = euclid_classify(data, threshold, unique_targets)
    return (predicted != gorund_truth).sum() / predicted.size
