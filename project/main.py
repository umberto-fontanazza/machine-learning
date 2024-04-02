import numpy as np
from csv import reader as csv_reader
from matplotlib import pyplot as plt

FEATURES_COUNT = 6
IMAGE_FOLDER = '/Users/umbertofontanazza/Desktop/'

def load(file_path: str):
    dataset = np.array([])
    labels = []
    with open(file_path) as csv_file:
        for row in csv_reader(csv_file):
            sample_values: list[float] = [float(value) for value in row[:-1]]
            sample_ndarr = np.array(sample_values)
            class_label = int(row[-1])
            dataset = np.concatenate((dataset, sample_ndarr))
            labels.append(class_label)
    labels = np.array(labels)
    dataset = dataset.reshape((len(dataset) // FEATURES_COUNT, FEATURES_COUNT)).T
    return dataset, labels

def visualize_single_attributes(dataset, labels):
    for attribute_index in range(FEATURES_COUNT):
        plt.figure(label = f'attribute [{attribute_index}]')
        for label, label_name in enumerate(('counterfeit', 'legit')):
            plt.hist(dataset[attribute_index, labels == label], alpha=.5,label=label_name, density=True)
        plt.legend()
        if attribute_index in (4, 5):
            plt.savefig(f'{IMAGE_FOLDER}attribute-[{attribute_index}]')
    # plt.show()

def save_scatter(dataset, labels):
    attributes = (4, 5)
    x_index, y_index = attributes
    plt.figure(label = f'Scatter attributes 4 and 5')
    for label, label_name in enumerate(('counterfeit', 'legit')):
        plt.scatter(dataset[x_index, labels == label], dataset[y_index, labels == label], label = label_name)
    plt.xlabel('Attribute 4')
    plt.ylabel('Attribute 5')
    plt.legend()
    plt.savefig(f'{IMAGE_FOLDER}scatter-attr-4-5')


def main():
    file_path = 'assets/trainData.csv'
    dataset, labels = load(file_path)
    # visualize_single_attributes(dataset, labels)
    save_scatter(dataset, labels)

if __name__ == '__main__':
    main()