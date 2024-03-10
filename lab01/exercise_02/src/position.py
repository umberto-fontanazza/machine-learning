from __future__ import annotations
from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def distance(self, other: Position) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)