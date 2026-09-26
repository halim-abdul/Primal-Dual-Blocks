# Architecture

- **src/primal_dual_blocks/** — reusable numerical core.
- **examples/** — small end-to-end programs.
- **experiments/** — reproducible benchmark scripts.
- **visualization/** — CSV-to-figure reporting.
- **tests/** — proximal, validation, and solver checks.
- **docs/** — mathematical assumptions and design decisions.

The solver accepts proximal maps as callables instead of hard-coding one objective. A future large-scale implementation can replace dense matrices with linear operators, sparse matrices, asynchronous workers, or GPU kernels while preserving the high-level API.
