from sklearn.datasets import load_iris
from numpy import cov
from numpy.linalg import eigh


def main():
    iris = load_iris()
    data, target = iris.data, iris.target
    data = data.T
    emp_cov = cov(data, bias=True)
    eigenvalues, eigenvectors = eigh(emp_cov)
    print(eigenvalues, eigenvectors)
    eig_1, eigv_1 = eigenvalues[0], eigenvectors[0]
    print(eig_1)


if __name__ == "__main__":
    main()
