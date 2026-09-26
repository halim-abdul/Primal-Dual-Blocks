"""Create a compact benchmark dashboard from CSV logs."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="artifacts/figures/benchmark_dashboard.png")
    a=p.parse_args()
    with Path(a.csv).open(newline="") as fh:
        rows=list(csv.DictReader(fh))

    blocks=[int(r["blocks"]) for r in rows]
    runtime=[float(r["seconds"]) for r in rows]
    objective=[float(r["final_objective"]) for r in rows]

    fig, axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(blocks,runtime,marker="o")
    axes[0].set_xlabel("Blocks"); axes[0].set_ylabel("Seconds"); axes[0].set_title("Runtime")
    axes[1].plot(blocks,objective,marker="o")
    axes[1].set_xlabel("Blocks"); axes[1].set_ylabel("Final objective"); axes[1].set_title("Optimization quality")
    fig.tight_layout()

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    print(out)


if __name__=="__main__":
    main()
