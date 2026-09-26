"""Proximal operators used by the examples and tests."""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft-thresholding."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def prox_l1(x: np.ndarray, step: float, lam: float) -> np.ndarray:
    """prox_{step * lam ||.||_1}(x)."""
    if step < 0 or lam < 0:
        raise ValueError("step and lam must be non-negative")
    return soft_threshold(x, step * lam)


def prox_squared_l2_conjugate(
    y: np.ndarray,
    sigma: float,
    b: np.ndarray,
) -> np.ndarray:
    r"""Prox of sigma * h* for h(z)=0.5||z-b||_2^2.

    Since h*(y)=0.5||y||^2 + <y,b>, the closed-form proximal map is
    (y - sigma*b)/(1+sigma).
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return (np.asarray(y, dtype=float) - sigma * np.asarray(b, dtype=float)) / (1.0 + sigma)
