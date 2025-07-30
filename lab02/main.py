from numpy import array, float64, int8, cov, diagonal
from pathlib import Path
from matplotlib.pyplot import scatter, hist, figure, show, legend, xlabel, ylabel
from csv import reader
from itertools import product
from ..lib.types import FloatMatrix, IntArray

IRIS_LABELS: list[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
IRIS_FEATURES: list[str] = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width",
]


def load_iris() -> tuple[FloatMatrix, IntArray]:
    accumulator: list[float] = []
    labels: list[int] = []
    with open(Path("iris.csv"), "r") as csv_file:
        csv_reader = reader(csv_file)
        for line in csv_reader:
            accumulator.extend([float(x) for x in line[:-1]])
            labels.append(IRIS_LABELS.index(line[-1]))
    data = array(accumulator, dtype=float64).reshape((150, 4)).T.copy()
    target = array(labels, dtype=int8)
    assert data.shape == (4, 150)
    assert target.shape == (150,)
    return data, target


def plot_hist(data: FloatMatrix, target: IntArray, feature: int):
    assert feature in range(len(IRIS_FEATURES))
    figure()
    for t in range(len(IRIS_LABELS)):
        hist(data[feature, target == t], density=True, label=IRIS_LABELS[t])
    legend()
    show()


def plot_scatter(data: FloatMatrix, target: IntArray, x_feature: int, y_feature: int):
    assert x_feature in range(data.shape[0])
    assert y_feature in range(data.shape[0])
    assert x_feature != y_feature

    figure()
    xlabel(IRIS_FEATURES[x_feature])
    ylabel(IRIS_FEATURES[y_feature])
    for t in range(len(IRIS_LABELS)):
        scatter(
            data[x_feature, target == t],
            data[y_feature, target == t],
            label=IRIS_LABELS[t],
        )
    legend()
    show()


def visualize_dataset(data: FloatMatrix, target: IntArray):
    for x_feature, y_feature in product(range(data.shape[0]), repeat=2):
        if x_feature == y_feature:
            plot_hist(data, target, x_feature)
        else:
            plot_scatter(data, target, x_feature, y_feature)


def statistics(data: FloatMatrix, target: IntArray):
    mean = data.mean(axis=1).reshape(data.shape[0], 1)
    print("Mean: \n", mean)
    covariance_m = cov(data, bias=True)
    print("Covariance matrix: \n", covariance_m)
    variance = diagonal(covariance_m).reshape(4, 1)
    print("Variance: \n", variance)


def main():
    data, target = load_iris()


if __name__ == "__main__":
    main()
