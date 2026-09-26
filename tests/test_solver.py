import numpy as np
import pytest
from primal_dual_blocks.problems import make_lasso_problem
from primal_dual_blocks.solver import BlockPrimalDual

def setup():
    p=make_lasso_problem(m=30,n=40,sparsity=4,blocks=4,seed=3)
    return p,BlockPrimalDual(p["K"],p["prox_g_blocks"],p["prox_h_conj"],p["blocks"],seed=3)
def test_solver_runs():
    p,s=setup(); r=s.run(np.zeros(40),np.zeros(30),40,objective=p["objective"])
    assert len(r.history["objective"])==40 and np.isfinite(r.x).all()
def test_bad_shape():
    p,s=setup()
    with pytest.raises(ValueError): s.run(np.zeros(39),np.zeros(30),2,objective=p["objective"])
