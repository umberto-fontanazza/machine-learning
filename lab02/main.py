from numpy.typing import NDArray
from csv import reader as csv_reader
import matplotlib.pyplot as plt
import numpy as np

class_labels = {
    'Iris-setosa': 0,
    'Iris-versicolor' : 1,
    'Iris-virginica' : 2,
}

def load(filename = 'iris.csv'): # missing type hint
    dataset = []
    labels = []
    with open(f'./assets/{filename}') as csv_file:
        for line in csv_reader(csv_file):
            sepal_length, sepal_width, petal_length, petal_width, label = line
            data_point = np.array([sepal_length, sepal_width, petal_length, petal_width])
            dataset.append(data_point.T)
            labels.append(class_labels[label])
    return (np.array(dataset).T, np.array(labels))

def visualize_single_attributes(dataset, labels):
    dataset_label0 = dataset[:, labels == 0]
    dataset_label1 = dataset[:, labels == 1]
    dataset_label2 = dataset[:, labels == 2]

    plt.figure(label='sepal_length')
    plt.hist(dataset_label0[0,:])
    plt.hist(dataset_label1[0,:])
    plt.hist(dataset_label2[0,:])
    plt.figure(label='sepal_width')
    plt.hist(dataset_label0[1,:])
    plt.hist(dataset_label1[1,:])
    plt.hist(dataset_label2[1,:])
    plt.figure(label='petal_length')
    plt.hist(dataset_label0[2,:])
    plt.hist(dataset_label1[2,:])
    plt.hist(dataset_label2[2,:])
    plt.figure(label='petal_width')
    plt.hist(dataset_label0[3,:])
    plt.hist(dataset_label1[3,:])
    plt.hist(dataset_label2[3,:])
    plt.show()

def visualize_scatter(dataset, labels):
    dataset_label0 = dataset[:, labels == 0]
    dataset_label1 = dataset[:, labels == 1]
    dataset_label2 = dataset[:, labels == 2]



def main():
    dataset, labels = load()
    # visualize_single_attributes(dataset, labels)
    visualize_scatter(dataset, labels)

if __name__ == '__main__':
    main()