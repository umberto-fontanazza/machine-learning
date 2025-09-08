from typing import Iterator


class BinaryApplication:
    pi: float
    cost_fn: float
    cost_fp: float

    def __init__(self, pi, cost_fn, cost_fp):
        self.pi = pi
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def __iter__(self) -> Iterator[float]:
        yield self.pi
        yield self.cost_fn
        yield self.cost_fp

    @property
    def effective_prior(self) -> float:
        pi, cost_fn, cost_fp = self
        return pi * cost_fn / (pi * cost_fn + (1 - pi) * cost_fp)


TEST_APPLICATIONS: list[BinaryApplication] = [
    BinaryApplication(*t)
    for t in [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1), (0.5, 1, 9), (0.5, 9, 1)]
]
