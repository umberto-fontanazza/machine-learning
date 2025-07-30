from pathlib import Path

from lib import load_from_csv
from lib.types import BoolArray, F64Matrix


def main():
    data: F64Matrix
    target: BoolArray
    data, target = load_from_csv(Path(Path(__file__).parent, "train_data.csv"))


if __name__ == "__main__":
    main()
