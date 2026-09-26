from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np
from primal_dual_blocks.problems import make_lasso_problem
from primal_dual_blocks.solver import BlockPrimalDual

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--iterations",type=int,default=1200)
    p.add_argument("--block-counts",type=int,nargs="+",default=[1,4,8,16])
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--output",default="artifacts/benchmark.csv")
    a=p.parse_args(); rows=[]
    for nb in a.block_counts:
        problem=make_lasso_problem(blocks=nb,seed=a.seed)
        solver=BlockPrimalDual(problem["K"],problem["prox_g_blocks"],problem["prox_h_conj"],problem["blocks"],seed=a.seed)
        t=time.perf_counter()
        result=solver.run(np.zeros(problem["K"].shape[1]),np.zeros(problem["K"].shape[0]),a.iterations,objective=problem["objective"],log_every=max(1,a.iterations//200))
        rows.append({"blocks":nb,"iterations":a.iterations,"seconds":time.perf_counter()-t,"final_objective":result.history["objective"][-1],"final_update_norm":result.history["update_norm"][-1]})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(out)
if __name__=="__main__": main()
