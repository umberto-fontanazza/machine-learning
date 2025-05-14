from csv import reader
from pprint import pprint
from numpy import array, concatenate

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


def main():
    dataset, labels = load_iris()


if __name__ == "__main__":
    main()
