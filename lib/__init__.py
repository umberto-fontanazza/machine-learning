from csv import reader
from pathlib import Path

from numpy import array
from numpy import bool as npBool
from numpy import float64

from .types import BoolArray, FloatMatrix, IntArray


def load_data_csv(path: Path) -> tuple[FloatMatrix, BoolArray | IntArray]:
    accumulator: list[float] = []
    labels: list[bool] = []
    n_features: int = 0
    with open(path, "r") as csv_file:
        csv_reader = reader(csv_file)
        for index, line in enumerate(csv_reader):
            if index == 0:
                n_features = len(line) - 1
            accumulator.extend([float(x) for x in line[:-1]])
            labels.append(bool(line[-1]))
    n_samples = len(accumulator) // n_features
    data = array(accumulator, dtype=float64)
    data: FloatMatrix = data.reshape(n_samples, n_features).T.copy()
    target: BoolArray = array(labels, dtype=npBool)
    return data, target
