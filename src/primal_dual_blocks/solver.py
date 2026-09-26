"""Readable randomized block-coordinate primal-dual solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
ProxBlock = Callable[[Array, float], Array]
ProxDual = Callable[[Array, float], Array]
Objective = Callable[[Array], float]


@dataclass
class SolverResult:
    x: Array
    y: Array
    history: dict[str, list[float]]


class BlockPrimalDual:
    """Randomized block-coordinate primal-dual splitting.

    Parameters
    ----------
    K:
        Dense matrix representation of the linear map. The code is intentionally
        explicit; replacing it by a LinearOperator is a natural extension.
    prox_g_blocks:
        One proximal callable per primal block. Signature: prox(v, tau).
    prox_h_conj:
        Proximal callable for h* with signature prox(y, sigma).
    blocks:
        Sequence of integer index arrays defining a partition of primal coordinates.
    tau, sigma:
        Optional step sizes. If omitted, conservative global values derived from
        ||K||_2 are used.
    theta:
        Extrapolation parameter.
    seed:
        RNG seed for randomized block selection.
    """

    def __init__(
        self,
        K: Array,
        prox_g_blocks: Sequence[ProxBlock],
        prox_h_conj: ProxDual,
        blocks: Sequence[Array],
        tau: float | None = None,
        sigma: float | None = None,
        theta: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.K = np.asarray(K, dtype=float)
        if self.K.ndim != 2:
            raise ValueError("K must be a 2D matrix")

        self.blocks = [np.asarray(block, dtype=int) for block in blocks]
        self._validate_partition()

        if len(prox_g_blocks) != len(self.blocks):
            raise ValueError("prox_g_blocks must contain one callable per block")
        self.prox_g_blocks = list(prox_g_blocks)
        self.prox_h_conj = prox_h_conj
        self.theta = float(theta)
        self.rng = np.random.default_rng(seed)

        norm_k = float(np.linalg.norm(self.K, 2))
        scale = max(norm_k, 1e-12)
        default_step = 0.99 / scale
        self.tau = float(default_step if tau is None else tau)
        self.sigma = float(default_step if sigma is None else sigma)

        if self.tau <= 0 or self.sigma <= 0:
            raise ValueError("tau and sigma must be positive")
        if self.tau * self.sigma * norm_k**2 >= 1.0 + 1e-12:
            raise ValueError("Require tau * sigma * ||K||_2^2 < 1 for baseline solver")

    def _validate_partition(self) -> None:
        n = self.K.shape[1]
        if not self.blocks:
            raise ValueError("at least one block is required")
        joined = np.concatenate(self.blocks)
        if joined.size != n or set(joined.tolist()) != set(range(n)):
            raise ValueError("blocks must form a partition of coordinates 0..n-1")

    def run(
        self,
        x0: Array,
        y0: Array,
        n_iter: int,
        objective: Objective | None = None,
        log_every: int = 1,
    ) -> SolverResult:
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if log_every < 1:
            raise ValueError("log_every must be >= 1")

        x = np.asarray(x0, dtype=float).copy()
        y = np.asarray(y0, dtype=float).copy()
        if x.shape != (self.K.shape[1],):
            raise ValueError("x0 has incompatible shape")
        if y.shape != (self.K.shape[0],):
            raise ValueError("y0 has incompatible shape")

        x_bar = x.copy()
        history: dict[str, list[float]] = {
            "iteration": [],
            "objective": [],
            "residual_norm": [],
            "update_norm": [],
            "block": [],
        }

        for k in range(1, n_iter + 1):
            y = self.prox_h_conj(y + self.sigma * (self.K @ x_bar), self.sigma)

            block_id = int(self.rng.integers(len(self.blocks)))
            idx = self.blocks[block_id]
            x_old = x.copy()

            grad_block = self.K[:, idx].T @ y
            trial = x[idx] - self.tau * grad_block
            x[idx] = self.prox_g_blocks[block_id](trial, self.tau)

            x_bar = x + self.theta * (x - x_old)

            if k % log_every == 0 or k == 1 or k == n_iter:
                update_norm = float(np.linalg.norm(x - x_old))
                residual_norm = float(np.linalg.norm(self.K @ x))
                history["iteration"].append(float(k))
                history["objective"].append(
                    float(objective(x)) if objective is not None else float("nan")
                )
                history["residual_norm"].append(residual_norm)
                history["update_norm"].append(update_norm)
                history["block"].append(float(block_id))

        return SolverResult(x=x, y=y, history=history)
