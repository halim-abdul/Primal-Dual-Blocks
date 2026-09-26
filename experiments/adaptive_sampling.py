"""Adaptive block-sampling prototype.

Blocks with larger recent update magnitudes receive higher sampling probability.
This is an experimental heuristic; benchmark and theoretical validation are required.
"""

from __future__ import annotations
import numpy as np


class AdaptiveSampler:
    def __init__(self, n_blocks: int, momentum: float = 0.9, floor: float = 1e-3, seed: int = 0):
        self.scores = np.ones(n_blocks, dtype=float)
        self.momentum = momentum
        self.floor = floor
        self.rng = np.random.default_rng(seed)

    def update(self, block: int, magnitude: float) -> None:
        target = max(float(magnitude), self.floor)
        self.scores[block] = self.momentum * self.scores[block] + (1.0 - self.momentum) * target

    def probabilities(self) -> np.ndarray:
        p = np.maximum(self.scores, self.floor)
        return p / p.sum()

    def sample(self) -> int:
        return int(self.rng.choice(len(self.scores), p=self.probabilities()))
