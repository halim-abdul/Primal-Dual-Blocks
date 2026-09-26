"""Synthetic convex problems for examples and reproducible experiments."""

from __future__ import annotations

import numpy as np

from .prox import prox_l1, prox_squared_l2_conjugate


def contiguous_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    if n_blocks < 1 or n_blocks > n:
        raise ValueError("n_blocks must be between 1 and n")
    return [arr.astype(int) for arr in np.array_split(np.arange(n), n_blocks)]


def make_lasso_problem(
    m: int = 120,
    n: int = 300,
    sparsity: int = 18,
    lam: float = 0.05,
    noise: float = 0.02,
    blocks: int = 8,
    seed: int = 7,
) -> dict:
    """Create min_x 0.5||Kx-b||^2 + lam||x||_1 with known sparse signal."""
    if sparsity > n:
        raise ValueError("sparsity cannot exceed n")

    rng = np.random.default_rng(seed)
    K = rng.normal(size=(m, n)) / np.sqrt(m)

    x_true = np.zeros(n)
    support = rng.choice(n, size=sparsity, replace=False)
    x_true[support] = rng.normal(size=sparsity)
    b = K @ x_true + noise * rng.normal(size=m)

    partition = contiguous_blocks(n, blocks)
    prox_blocks = [
        (lambda z, tau, lam=lam: prox_l1(z, tau=tau, lam=lam))
        for _ in partition
    ]

    def prox_h_conj(y: np.ndarray, sigma: float) -> np.ndarray:
        return prox_squared_l2_conjugate(y, sigma=sigma, b=b)

    def objective(x: np.ndarray) -> float:
        residual = K @ x - b
        return 0.5 * float(residual @ residual) + lam * float(np.abs(x).sum())

    return {
        "K": K,
        "b": b,
        "x_true": x_true,
        "blocks": partition,
        "prox_g_blocks": prox_blocks,
        "prox_h_conj": prox_h_conj,
        "objective": objective,
        "lam": lam,
    }
