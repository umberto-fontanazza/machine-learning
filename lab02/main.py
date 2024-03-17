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
        covariance = np.zeros((ndarray.shape[0], ndarray.shape[0]))
        for sample in ndarray.T:
            sample = sample.reshape((sample.shape[0], 1))
            print(sample.shape, mean.shape)
            partial = (sample - mean) * (sample - mean).T
            covariance += partial
        return covariance / n
    elif strategy == '@':
        centered_data = ndarray - mean
        return centered_data @ centered_data.T / n
    else:
        raise ValueError(f'Invalid strategy {strategy}')

def print_statistics(dataset: NDArray):
    mean: NDArray = dataset.mean(1)
    variance: NDArray = dataset.var(1)
    standard_deviation: NDArray = dataset.std(1)
    covariance_matrix: NDArray = empyrical_covariance(dataset)

    print(f'Mean:               {mean}')
    print(f'Variance:           {variance}')
    print(f'Standard deviation: {standard_deviation}')
    print(f'Covariance matrix:  {covariance_matrix}')

def statistics(dataset: NDArray, labels):
    print('Entire dataset:')
    print_statistics(dataset)

    for class_label, class_index in class_labels.items():
        print(f'{class_label} [{class_index}]')
        print_statistics(dataset[:, labels == class_index])
        print()

def main():
    np.set_printoptions(precision=1, sign=' ')
    dataset, labels = load()
    # visualize_single_attributes(dataset, labels)
    # visualize_scatter(dataset, labels)
    statistics(dataset, labels)

if __name__ == '__main__':
    main()