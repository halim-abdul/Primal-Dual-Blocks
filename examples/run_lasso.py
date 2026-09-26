from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from primal_dual_blocks.problems import make_lasso_problem
from primal_dual_blocks.solver import BlockPrimalDual

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--iterations",type=int,default=1500)
    p.add_argument("--blocks",type=int,default=8)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--output",default="artifacts/lasso_history.csv")
    a=p.parse_args()
    problem=make_lasso_problem(blocks=a.blocks,seed=a.seed)
    solver=BlockPrimalDual(problem["K"],problem["prox_g_blocks"],problem["prox_h_conj"],problem["blocks"],seed=a.seed)
    result=solver.run(np.zeros(problem["K"].shape[1]),np.zeros(problem["K"].shape[0]),a.iterations,objective=problem["objective"])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    keys=list(result.history)
    with out.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader()
        for row in zip(*(result.history[k] for k in keys)): w.writerow(dict(zip(keys,row)))
    print(f"final objective: {result.history['objective'][-1]:.6f}")
    print(f"recovery error: {np.linalg.norm(result.x-problem['x_true']):.6f}")
    print(f"history: {out}")
if __name__=="__main__": main()
