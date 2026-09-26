"""Utilities for experimental multi-block Jacobi-style updates."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def sample_distinct_blocks(rng: np.random.Generator, n_blocks: int, batch_size: int) -> np.ndarray:
    if not 1 <= batch_size <= n_blocks:
        raise ValueError("batch_size must be between 1 and n_blocks")
    return np.asarray(rng.choice(n_blocks, size=batch_size, replace=False), dtype=int)


def block_union(blocks: Sequence[np.ndarray], selected: Sequence[int]) -> np.ndarray:
    return np.concatenate([np.asarray(blocks[i], dtype=int) for i in selected])


def batched_gradient(K: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return K[:, indices].T @ y
