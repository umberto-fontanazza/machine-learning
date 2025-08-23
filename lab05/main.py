from matplotlib.pyplot import figure, plot, show
from numpy import abs, array, cov, dtype, exp, float64, linspace, load, log, ndarray, pi
from numpy.linalg import inv, slogdet


def log_pdf_normal(x, mu, C) -> ndarray[tuple[int], dtype[float64]]:
    M, _ = x.shape
    assert M == mu.shape[0]
    assert M == C.shape[0]
    assert M == C.shape[1]

    x_center = x - mu
    return -0.5 * (
        M * log(2 * pi)
        + slogdet(C)[1]
        + ((x_center.T @ inv(C)).T * x_center).sum(axis=0)
    )


def main():
    x = load("solution/XND.npy")
    mu, C = x.mean(axis=1).reshape(x.shape[0], 1), cov(x, bias=True)
    ll = log_pdf_normal(x, mu, C).sum()
    print(f"{ll=}")


if __name__ == "__main__":
    main()
