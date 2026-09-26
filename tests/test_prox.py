import numpy as np
from primal_dual_blocks.prox import prox_l1, prox_squared_l2_conjugate, soft_threshold

def test_soft_threshold():
    x=np.array([-2.0,-0.25,0.0,0.5,3.0])
    assert np.allclose(soft_threshold(x,0.5),[-1.5,0,0,0,2.5])
def test_l1_prox():
    assert np.allclose(prox_l1(np.array([1.0,-1.0]),0.5,0.2),[0.9,-0.9])
def test_dual_prox():
    y=np.array([2.0,-1.0]); b=np.array([1.0,3.0])
    assert np.allclose(prox_squared_l2_conjugate(y,0.5,b),(y-0.5*b)/1.5)
