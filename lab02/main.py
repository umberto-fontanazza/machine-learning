from csv import reader
from numpy import array, concatenate, ndarray
from matplotlib import pyplot as plt
from itertools import product

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]


def load_iris():
    dataset, labels = [], []
    with open("./iris.csv", "r") as file:
        csv_file = reader(file)
        for row in csv_file:
            features, label = row[0:4], row[4]
            features, label = [float(value) for value in features], class_names.index(
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
                label=f"feature {feature}, {class_names[class_label]}",
                density=True,
            )
        plt.title(f"Feature {feature}")
        plt.legend()
        plt.show()


def feature_pairs_visualization(dataset, labels):
    for feature_1, feature_2 in product(range(4), range(4)):
        if feature_1 == feature_2:
            continue
        for class_label in range(3):
            class_mask = labels == class_label
            plt.scatter(dataset[feature_1, class_mask], dataset[feature_2, class_mask])
        plt.xlabel(feature_names[feature_1])
        plt.ylabel(feature_names[feature_2])
        plt.show()


def stats(dataset: ndarray):
    mean: ndarray = dataset.mean(1).reshape(dataset.shape[0], 1)
    centered_data = dataset - mean
    emp_covariance = centered_data @ centered_data.T / centered_data.shape[1]
    print(emp_covariance)
    # faster alternative print(cov(dataset, bias=True))


def main():
    dataset, labels = load_iris()


if __name__ == "__main__":
    main()
