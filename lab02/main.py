import numpy as np
from numpy import array, ndarray, dtype, float64, int8
from typing import Literal
from pathlib import Path
from csv import reader

LABELS: list[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris() -> tuple[
    ndarray[tuple[int, int], dtype[float64]],  # data
    ndarray[tuple[int], dtype[int8]],  # target
]:
    accumulator: list[float] = []
    labels: list[int] = []
    with open(Path("iris.csv"), "r") as csv_file:
        csv_reader = reader(csv_file)
        for line in csv_reader:
            accumulator.extend([float(x) for x in line[:-1]])
            labels.append(LABELS.index(line[-1]))
    data = array(accumulator, dtype=float64).reshape((150, 4)).T.copy()
    target = array(labels, dtype=int8)
    assert data.shape == (4, 150)
    assert target.shape == (150,)
    return data, target


def main():
    data, target = load_iris()


if __name__ == "__main__":
    main()
