"""Independent causal M3 v2 integration proof before interpreting rerun."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h'
sys.path.insert(0,str(AREA/'integration'))
import paired_pilot as p
p.PROTOCOL_FILE=AREA/'manifests/pilot-v2.json';p.PROTOCOL=json.loads(p.PROTOCOL_FILE.read_text())
Model,torch=p.model_classes()

class Wiring(unittest.TestCase):
    def test_M3_stores_structured_pre_read_state_and_gradient_path(self):
        torch.manual_seed(112);m=Model('M3',11);x=torch.randn(1,4,8,requires_grad=True)
        ev=[[{'kind':'SET','entity':0,'operator':None,'value':1},
             {'kind':'OP','entity':0,'operator':1,'value':None},
             {'kind':'QUERY','entity':0,'operator':None,'value':None},
             {'kind':'NOOP','entity':0,'operator':None,'value':None}]]
        pre=[];actual=m.memory.step
        def track(z,state,control):pre.append(z);return actual(z,state,control)
        m.memory.step=track;m(x,ev)
        ast=x[:,0]
        expected,_,_=m.algebra.step(x[:,1],ast,control={'op_id':torch.tensor([1])})
        torch.testing.assert_close(pre[1],expected,rtol=0,atol=0)
        self.assertGreater(float((pre[1]-x[:,1]).norm()),.01)
        # Inspect committed values alone, excluding readout/direct algebra path.
        ms=m.memory.initial_state(1)
        from research.mamba3h.memory.slots import Control
        _,ms,_=actual(pre[1],ms,Control(torch.tensor([True]),torch.tensor([False]),torch.tensor([0])))
        ms.values.square().sum().backward()
        self.assertIsNotNone(m.algebra.u.grad);self.assertGreater(float(m.algebra.u.grad.norm()),0)
        self.assertGreater(float(x.grad[:,0].norm()),0)

    def test_no_future_target_or_same_step_retrieval_in_write_features(self):
        m=Model('M3',23);torch.manual_seed(21);x=torch.randn(1,3,8)
        ev=[[{'kind':'SET','entity':0,'operator':None,'value':1},
             {'kind':'OP','entity':0,'operator':0,'value':None},
             {'kind':'QUERY','entity':0,'operator':None,'value':None}]]
        altered=x.clone();altered[:,2:]+=100
        y,_=m(x,ev);z,_=m(altered,ev);torch.testing.assert_close(y[:,:2],z[:,:2],rtol=0,atol=0)
        self.assertEqual(p.PROTOCOL['hyperparameter_changes'],{})

if __name__=='__main__':unittest.main(verbosity=2)
