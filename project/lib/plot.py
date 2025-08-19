from itertools import product
from typing import Literal

from matplotlib.pyplot import figure, hist, legend, scatter, show, title, xlabel, ylabel

from .types import BoolArray, F64Matrix

LABELS = ["counterfeit", "genuine"]


def plot_hist(data: F64Matrix, target: BoolArray, feature: int):
    figure()
    title(f"Feature {feature}")
    for t in [True, False]:
        hist(data[feature, target == t], density=True, label=LABELS[t], alpha=0.4)
    legend()
    show()


def plot_scatter(data: F64Matrix, target: BoolArray, x_feature: int, y_feature: int):
    assert x_feature in range(data.shape[0])
    assert y_feature in range(data.shape[0])
    assert x_feature != y_feature

    figure()
    xlabel(f"Feature {x_feature}")
    ylabel(f"Feature {y_feature}")
    for cls in [True, False]:
        scatter(
            data[x_feature, target == cls],
            data[y_feature, target == cls],
            label=LABELS[cls],
        )
    legend()
    show()


def visualize(
    data: F64Matrix, target: BoolArray, mode: Literal["all", "hist", "scatter"] = "all"
):
    for feature_1, feature_2 in product(range(data.shape[0]), repeat=2):
        if feature_1 > feature_2:
            continue
        elif feature_1 == feature_2 and mode != "scatter":
            plot_hist(data, target, feature_1)
        else:
            if mode != "hist":
                plot_scatter(data, target, feature_1, feature_2)
