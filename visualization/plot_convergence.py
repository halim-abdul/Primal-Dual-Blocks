from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_rows(path):
    with Path(path).open(newline="") as fh: return list(csv.DictReader(fh))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",required=True); p.add_argument("--output-dir",default="artifacts/figures"); a=p.parse_args()
    rows=read_rows(a.csv); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if "iteration" in rows[0]:
        x=[float(r["iteration"]) for r in rows]; y=[float(r["objective"]) for r in rows]
        plt.figure(); plt.plot(x,y); plt.xlabel("Iteration"); plt.ylabel("Objective"); plt.title("Primal-dual convergence"); plt.tight_layout(); plt.savefig(out/"objective_vs_iteration.png",dpi=180); plt.close()
    else:
        b=[int(r["blocks"]) for r in rows]; s=[float(r["seconds"]) for r in rows]; o=[float(r["final_objective"]) for r in rows]
        plt.figure(); plt.plot(b,s,marker="o"); plt.xlabel("Number of blocks"); plt.ylabel("Runtime (s)"); plt.title("Runtime scaling by block count"); plt.tight_layout(); plt.savefig(out/"runtime_vs_blocks.png",dpi=180); plt.close()
        plt.figure(); plt.plot(b,o,marker="o"); plt.xlabel("Number of blocks"); plt.ylabel("Final objective"); plt.title("Final objective by block count"); plt.tight_layout(); plt.savefig(out/"objective_vs_blocks.png",dpi=180); plt.close()
if __name__=="__main__": main()
