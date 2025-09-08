from numpy import average, cov, eye, float64, log, stack, uint8, unique
from scipy.special import logsumexp

from .normal import log_normal_density
from .types import F64Array, F64Matrix, U8Array


class Mvg:
    n_dim: int
    n_cls: int
    targets: U8Array  # the integer associated with the classes, since they might not start from 0
    cls_means: list[F64Matrix]
    cls_covs: list[F64Matrix]
    naive: bool
    tied: bool

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
        self.naive, self.tied = naive, tied

        if tied:
            self.cls_covs = [
                average(stack(self.cls_covs), axis=0, weights=cls_weights)
                for _ in range(self.n_cls)
            ]

        if naive:
            [c * eye(self.n_dim, self.n_dim) for c in self.cls_covs]

    def _log_posterior(self, data: F64Matrix, prior: F64Array) -> F64Matrix:
        log_cls_conditional = stack(
            [
                log_normal_density(data, mu, c)
                for mu, c in zip(self.cls_means, self.cls_covs)
            ]
        )
        log_joint = log_cls_conditional + log(prior.reshape(-1, 1))
        return log_joint - logsumexp(log_joint, axis=0)

    def score(self, data: F64Matrix, prior: F64Array) -> F64Array:
        """Works for binary case only"""
        assert prior.size == 2 == self.n_cls
        log_posterior = self._log_posterior(data, prior)
        return log_posterior[1, :] - log_posterior[0, :]

    def inference(self, data: F64Matrix, prior: F64Array) -> U8Array:
        log_posterior = self._log_posterior(data, prior)
        return log_posterior.argmax(axis=0).astype(uint8)

    @property
    def name(self) -> str:
        return f"Mvg {"naive " if self.naive else ""}{"tied " if self.tied else ""}"
