from typing import Literal
from numpy.typing import NDArray
from csv import reader as csv_reader
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

class_labels = {
    'Iris-setosa': 0,
    'Iris-versicolor' : 1,
    'Iris-virginica' : 2,
}

attribute_indexes = {
    'sepal_length': 0,
    'sepal_width': 1,
    'petal_length': 2,
    'petal_width': 3
}

def load(filename = 'iris.csv'): # missing type hint
    dataset = []
    labels = []
    with open(f'./assets/{filename}') as csv_file:
        for line in csv_reader(csv_file):
            sepal_length, sepal_width, petal_length, petal_width, label = line
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)
            data_point = np.array([sepal_length, sepal_width, petal_length, petal_width])
            dataset.append(data_point.T)
            labels.append(class_labels[label])
    return (np.array(dataset).T, np.array(labels))

def visualize_single_attributes(dataset, labels):
    for attribute, attribute_index in attribute_indexes.items():
        plt.figure(label=attribute)
        for label in class_labels.values():
            plt.hist(dataset[attribute_index, labels == label])
    plt.show()

def visualize_scatter(dataset, labels):
    # there are 4 attributes, hence there are 6 possible couples of attributes (unordered)
    for item1, item2 in combinations(attribute_indexes.items(), 2):
        attribute1, index1 = item1
        attribute2, index2 = item2
        plt.figure(label=f'{attribute1} vs {attribute2}')
        for label in class_labels.values():
            x, y = dataset[index1, labels == label], dataset[index2, labels == label]
            plt.scatter(x, y)
            plt.xlabel(attribute1)
            plt.ylabel(attribute2)

    plt.show()

def empyrical_covariance(ndarray: NDArray, strategy: Literal['fast','loop','@'] = 'fast'):
    mean = ndarray.mean(axis=1).reshape(ndarray.shape[0], 1)
    n = ndarray.shape[1]
    if strategy == 'fast':
        return np.cov(ndarray)
    elif strategy == 'loop':
        covariance = 0
        for sample in ndarray.T:
            sample = sample.reshape((sample.shape[0], 1))
            print(sample.shape, mean.shape)
            partial = (sample - mean) * (sample - mean).T
            covariance += partial
        return covariance / n
    elif strategy == '@':
        centered_data = ndarray - mean
        return centered_data @ centered_data.T / n

def statistics(dataset: NDArray):
    mean = dataset.mean(1).reshape(dataset.shape[0], 1)
    centered_data = dataset - mean
    print(dataset[:, :5])
    print(centered_data[:, :5])

def main():
    np.set_printoptions(precision=1, sign=' ')
    dataset, labels = load()
    # visualize_single_attributes(dataset, labels)
    # visualize_scatter(dataset, labels)
    # statistics(dataset)

if __name__ == '__main__':
    main()