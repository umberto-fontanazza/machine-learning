from matplotlib import pyplot as plt
from pathlib import Path
from numpy import array, concatenate
from csv import reader
from itertools import product

target_names = ["fake", "genuine"]


def load_data():
    data, target = [], []
    with open(Path(".", "train-data.csv"), "r") as csv_file:
        csv_file = reader(csv_file)
        for index, line in enumerate(csv_file):
            features, label = line[0:6], line[6]
            features = array([float(x) for x in features]).reshape(6, 1)
            data.append(features)
            target.append(int(label))
    data = concatenate(data, axis=1)
    target = array(target)
    return data, target


def single_feature_plot(data, target):
    for feature in range(6):
        for t in range(2):
            plt.hist(
                data[feature, target == t],
                label=f"Feature {feature} class {target_names[t]}",
                density=True,
            )
        plt.title(f"Feature {feature}")
        plt.legend()
        plt.show()


def feature_pair_plot(data, target):
    for feature_1, feature_2 in product(range(6), range(6)):
        if feature_1 == feature_2:
            continue
        for t in range(2):
            plt.scatter(data[feature_1, target == t], data[feature_2, target == t])
        plt.xlabel(f"Feature {feature_1}")
        plt.ylabel(f"Feature {feature_2}")
        # plt.legend()
        plt.title(f"Features {feature_1} over {feature_2}")
        plt.show()


def main():
    data, target = load_data()
    # single_feature_plot(data, target)
    # feature_pair_plot(data, target)


if __name__ == "__main__":
    main()
