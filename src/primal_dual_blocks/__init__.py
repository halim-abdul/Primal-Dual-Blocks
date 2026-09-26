"""Block-coordinate primal-dual optimization."""

from .solver import BlockPrimalDual, SolverResult
from .prox import soft_threshold, prox_l1, prox_squared_l2_conjugate

__all__ = [
    "BlockPrimalDual",
    "SolverResult",
    "soft_threshold",
    "prox_l1",
    "prox_squared_l2_conjugate",
]
