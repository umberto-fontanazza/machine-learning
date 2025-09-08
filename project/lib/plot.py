from itertools import product
from typing import Literal

from matplotlib.pyplot import figure, hist, legend, plot, scatter
from matplotlib.pyplot import show as plt_show
from matplotlib.pyplot import title, xlabel, xlim, ylabel, ylim
from numpy import array, exp, float64, linspace, unique
from sklearn.metrics import confusion_matrix

from .binary import cost_from_fn_fp
from .evaluation import dcf, dcf_min_bin
from .mvg import Mvg
from .types import BoolArray, F64Array, F64Matrix, U8Array

LABELS = ["counterfeit", "genuine"]


def plot_hist_u8(data: F64Array, target: U8Array):
    for t in unique(target):
        hist(data[target == t], alpha=0.4, label=f"{t}")
    legend()
    plt_show()


def plot_hist(data: F64Matrix, target: BoolArray, feature: int):
    figure()
    title(f"Feature {feature}")
    for t in [True, False]:
        hist(data[feature, target == t], density=True, label=LABELS[t], alpha=0.4)
    legend()
    plt_show()


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
    plt_show()


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


def plot_bayes_error(
    model: Mvg,
    data: F64Matrix,
    target: U8Array,
    interval: tuple[float, float] = (-4, 4),
    show=True,
):
    log_odds = linspace(interval[0], interval[1], 20)
    prior = 1 / (1 + exp(-log_odds))

    dcfs = []
    dcfs_min = []
    for pi in prior:
        prior = array([1 - pi, pi], dtype=float64)
        scores = model.score(data, prior)
        predicted = model.inference(data, prior)
        conf_m = confusion_matrix(target, predicted).T
        cost = cost_from_fn_fp(1, 1)
        dcfs.append(dcf(conf_m, cost, prior))
        dcfs_min.append(dcf_min_bin(scores, target, cost, prior))
    title(f"{model.name} bayes error")
    plot(log_odds, dcfs, label="dcf", color="r")
    plot(log_odds, dcfs_min, label="dcf-min", color="b")
    xlim(interval[0], interval[1])
    ylim(0, 1.1)
    if show:
        plt_show()
