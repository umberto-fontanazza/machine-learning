# step 1 load the data, and the labels(target)
# step 2 compute covariance matrix
# step 3 use singular value decomposition to get the eigenvectors
# step 4 extract the change of basis matrix
# step 5 project the data

from numpy.typing import NDArray
from sklearn.datasets import load_iris
from numpy import linalg
from matplotlib import pyplot as plt
import numpy as np

# TODO: type hint NDArray shapes
def get_basis_change_matrix(dataset: NDArray, desired_dimensions: int):
    covariance_matrix = np.cov(dataset, bias=True)
    U, _, _ = linalg.svd(covariance_matrix)
    return U[:, :desired_dimensions]

def main():
    iris = load_iris()
    data, target = iris.data.T, iris.target # pyright: ignore see https://stackoverflow.com/questions/78287188/pylance-complaining-for-sklearn-datasets-load-iris
    P_2_dimensions = get_basis_change_matrix(data, 2)
    projected_data_2_dimensions = P_2_dimensions.T @ data

    # plot scatter and compare with lab figure
    plt.figure()
    for label in range(3):
        x, y = projected_data_2_dimensions[0, target == label], projected_data_2_dimensions[1, target == label]
        plt.scatter(x, y)
    plt.show()
    

if __name__ == '__main__':
    main()