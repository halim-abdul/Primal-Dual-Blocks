"""Run a reproducible scaling sweep over problem dimensions and block counts."""

from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from primal_dual_blocks.problems import make_lasso_problem
from primal_dual_blocks.solver import BlockPrimalDual


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dimensions",type=int,nargs="+",default=[200,500,1000])
    p.add_argument("--blocks",type=int,nargs="+",default=[1,4,8,16])
    p.add_argument("--iterations",type=int,default=500)
    p.add_argument("--output",default="artifacts/scaling_study.csv")
    a=p.parse_args(); rows=[]

    for n in a.dimensions:
        m=max(40,n//3)
        for nb in a.blocks:
            if nb>n: continue
            problem=make_lasso_problem(m=m,n=n,sparsity=max(5,n//50),blocks=nb,seed=11)
            solver=BlockPrimalDual(problem["K"],problem["prox_g_blocks"],problem["prox_h_conj"],problem["blocks"],seed=11)
            start=time.perf_counter()
            result=solver.run(np.zeros(n),np.zeros(m),a.iterations,objective=problem["objective"],log_every=a.iterations)
            rows.append({"n":n,"m":m,"blocks":nb,"iterations":a.iterations,"seconds":time.perf_counter()-start,"objective":result.history["objective"][-1]})

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(out)


if __name__=="__main__":
    main()
