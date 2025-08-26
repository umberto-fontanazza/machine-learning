from numpy import average, cov, eye, float64, log, ones, stack, uint8, unique
from sklearn.preprocessing import normalize

from .normal import log_normal_density
from .types import F64Matrix, U8Array


class Mvg:
    n_dim: int
    n_cls: int
    targets: U8Array  # the integer associated with the classes, since they might not start from 0
    cls_means: list[F64Matrix]
    cls_covs: list[F64Matrix]

    def __init__(
        self, train_data: F64Matrix, train_target: U8Array, naive=False, tied=False
    ):
        self.targets, cls_weights = unique(train_target, return_counts=True)
        self.n_dim, self.n_cls = train_data.shape[0], len(self.targets)
        self.cls_means = [
            train_data[:, train_target == t].mean(axis=1).reshape(-1, 1)
            for t in self.targets
        ]
        self.cls_covs = [
            cov(train_data[:, train_target == t], bias=True, dtype=float64)
            for t in self.targets
        ]

        if tied:
            self.cls_covs = [
                average(stack(self.cls_covs), axis=0, weights=cls_weights)
                for _ in range(self.n_cls)
            ]

        if naive:
            [c * eye(self.n_dim, self.n_dim) for c in self.cls_covs]

    def inference(
        self, data: F64Matrix, ground_truth: U8Array | None = None, prior="uniform"
    ) -> U8Array | float64:
        """Returns an array with inferred classes or the error rate if the ground_truth is provided"""
        cls_conditional_ll = stack(
            [
                log_normal_density(data, mu, c)
                for (mu, c) in zip(self.cls_means, self.cls_covs)
            ]
        )
        ll_ratio = cls_conditional_ll[1, :] - cls_conditional_ll[0, :]
        prior = normalize(
            ones((self.n_cls, 1), dtype=float64), norm="l1", axis=0, copy=False
        )
        log_prior = log(prior)
        threshold = log_prior[1, :] - log_prior[0, :]
        predicted = ll_ratio > threshold
        if ground_truth is None:
            return predicted.astype(uint8)
        err_rate = (predicted != ground_truth).sum() / ground_truth.size
        return err_rate
