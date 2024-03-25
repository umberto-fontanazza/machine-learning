import numpy as np
from csv import reader as csv_reader

FEATURES_COUNT = 6

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

def main():
    file_path = 'assets/trainData.csv'
    dataset, labels = load(file_path)

if __name__ == '__main__':
    main()