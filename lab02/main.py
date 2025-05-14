from csv import reader
from pprint import pprint
from numpy import array, concatenate
from matplotlib import pyplot as plt
from itertools import product

class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris():
    dataset, labels = [], []
    with open("./iris.csv", "r") as file:
        csv_file = reader(file)
        for row in csv_file:
            features, label = row[0:4], row[4]
            features, label = [float(value) for value in features], class_labels.index(
                label
            )
            dataset.append(array(features).reshape(4, 1))
            labels.append(label)
        dataset = concatenate(dataset, axis=1)
    return dataset, array(labels)


def features_per_class(dataset, labels):
    for feature in range(4):
        for class_label in range(3):
            plt.hist(
                dataset[feature, labels == class_label],
                label=f"feature {feature}, {class_labels[class_label]}",
                density=True,
            )
        plt.title(f"Feature {feature}")
        plt.show()
    plt.legend()


def main():
    dataset, labels = load_iris()
    # features_per_class(dataset, labels)


if __name__ == "__main__":
    main()
