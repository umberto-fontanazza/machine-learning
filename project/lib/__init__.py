from csv import reader
from pathlib import Path

from lib.types import BoolArray, F64Matrix
from numpy import array
from numpy import bool as npBool
from numpy import float64


def load_from_csv(path: Path) -> tuple[F64Matrix, BoolArray]:
    buffer: list[float] = []
    labels: list[bool] = []
    n_features: int = 0
    with open(path, "r") as csv_file:
        csv_reader = reader(csv_file)
        for index, line in enumerate(csv_reader):
            if index == 0:
                n_features = len(line) - 1
            buffer.extend([float(x) for x in line[:-1]])
            labels.append(int(line[-1]) != 0)
    n_samples: int = len(buffer) // n_features
    data: F64Matrix = array(buffer, dtype=float64)
    data = data.reshape(n_samples, n_features).T.copy()
    target: BoolArray = array(labels, dtype=npBool)
    return data, target
