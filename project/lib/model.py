from numpy import float64, unique

from .classifier import euclid_classify, euclid_threshold
from .lda import get_lda_lt
from .pca import get_pca_lt
from .types import F64Matrix, U8Array


class PCA_LDA_euclid_binary:
    """This model applies PCA as dimensionality reduction technique reducing the dimensions to pca_m then finds a single
    LDA direction on which a simple euclidean classifier is used."""

    def __init__(self, data: F64Matrix, target: U8Array, pca_m: int):
        self.unique_targets = unique(target)
        self.P = get_pca_lt(data, pca_m)
        data = self.P.T @ data
        self.W = get_lda_lt(data, target, 1)
        if pca_m == 2:
            self.W = -self.W
        projected_data = (self.W.T @ data).flatten()
        self.threshold = euclid_threshold(projected_data, target)

    def inference(
        self, data: F64Matrix, ground_truth: U8Array | None
    ) -> U8Array | float64:
        """Returns an array with inferred classes or the error rate if the ground_truth is provided"""
        data = self.P.T @ data
        data = self.W.T @ data
        # check mean of class one is right of mean of class 0
        predicted = euclid_classify(data.flatten(), self.threshold, self.unique_targets)
        if ground_truth is None:
            return predicted
        else:
            return (predicted != ground_truth).sum() / ground_truth.size

    @property
    def pca_m(self) -> int:
        return self.P.shape[1]
