from pathlib import Path

from lib import load_from_csv
from lib.pca import get_pca_lt
from lib.plot import visualize
from lib.types import BoolArray, F64Matrix


def main():
    data: F64Matrix
    target: BoolArray
    data, target = load_from_csv(Path(Path(__file__).parent, "train-data.csv"))
    P = get_pca_lt(data, 6)
    data = P.T @ data
    visualize(data, target, mode="hist")


if __name__ == "__main__":
    main()
