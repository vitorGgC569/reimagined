"""Root2 independent integration/negative controls; completed snapshots only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
SNAP=ROOT/'research/mamba3h/integration/vendor'
if not (SNAP/'snapshot.json').exists():raise RuntimeError('Completed handoffs must be snapshotted before tests')
sys.path.insert(0,str(SNAP));sys.path.insert(1,str(ROOT));sys.path.insert(0,str(ROOT/'research/mamba3h/integration'))
import torch
torch.set_num_threads(2)
from research.mamba3h.algebra.operators import transition,StructuredTransition,compose,s5_matrices
from research.mamba3h.memory.slots import CausalSlotMemory,Control,OracleRoutes,route_oracle
from research.mamba3h.benchmarks.generation import generate
from research.mamba3h.benchmarks.schema import encode_numeric,model_view,validate_model_view
from paired_pilot import model_classes

class Independent(unittest.TestCase):
    def test_householder_noncommutative_and_homogeneous_contraction(self):
        torch.manual_seed(319)
        matrices=s5_matrices();self.assertGreater(float((matrices[0]@matrices[1]-matrices[1]@matrices[0]).norm()),1.)
        m=StructuredTransition(dim=8,rank=4,operations=10,seed=319)
        h=torch.randn(4,8,dtype=torch.float64);start=h.norm(dim=-1)
        for i in range(64):h=(m.matrices(torch.tensor([i%10]*4))@h.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(bool((h.norm(dim=-1)<=start+1e-10).all()))
        mc=StructuredTransition(dim=8,rank=4,operations=10,commuting=True,seed=319)
        a=mc.matrices(torch.tensor([0,1]));self.assertLess(float((a[0]@a[1]-a[1]@a[0]).norm()),1e-12)
        self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in mc.parameters()))

    def test_slot_no_read_and_entity_revocation_no_resurrection(self):
        torch.manual_seed(3);mem=CausalSlotMemory(8,2,1);s=mem.initial_state(1)
        def c(w=False,r=False,e=0,rev=-1):return Control(torch.tensor([w]),torch.tensor([r]),torch.tensor([e]),torch.tensor([rev]))
        x=torch.randn(1,8,requires_grad=True);_,s,st=mem.step(x,s,c(w=True))
        self.assertEqual(st['similarity_evaluations'],0);self.assertEqual(st['topk_calls'],0)
        other=torch.randn(1,8);y,s,st=mem.step(other,s,c(r=True,e=1));torch.testing.assert_close(y,other,rtol=0,atol=0)
        y,s,st=mem.step(other,s,c(r=True,rev=0));torch.testing.assert_close(y,other,rtol=0,atol=0)
        self.assertEqual(st['value_reads'],0);self.assertEqual(int(s.occupied.sum()),0)
        with self.assertRaises(TypeError):mem.step(other,s,{'write':False,'read':True,'answer':17})
        with self.assertRaises(TypeError):OracleRoutes(value=torch.ones(1))
        with self.assertRaises(ValueError):route_oracle('W',c(),OracleRoutes(read=torch.tensor([True])))

    def test_slot_gradients_from_past_not_future(self):
        torch.manual_seed(5);mem=CausalSlotMemory(8,2,1);s=mem.initial_state(1)
        x=torch.randn(1,8,requires_grad=True);q=torch.randn(1,8,requires_grad=True)
        _,s,_=mem.step(x,s,Control(torch.tensor([True]),torch.tensor([False]),torch.tensor([0])))
        y,_,_=mem.step(q,s,Control(torch.tensor([False]),torch.tensor([True]),torch.tensor([0])))
        y.square().sum().backward();self.assertGreater(float(x.grad.norm()),0);self.assertGreater(float(q.grad.norm()),0)
        self.assertGreater(float(mem.value.weight.grad.norm()),0)

    def test_targets_metadata_never_change_model_input(self):
        ep=generate('inst',11,'train',1,length=16,entities=3,operations=4,distractors=2,overwrite=1,revocations=1,queries=4)[0]
        altered=copy.deepcopy(ep);altered['targets']=[999]*len(ep['targets']);altered['metadata']={'answer':999,'future':'secret'}
        self.assertEqual(model_view(ep),model_view(altered));self.assertEqual(encode_numeric(ep),encode_numeric(altered))
        bad=model_view(ep);bad['events'][0]['answer']=42
        with self.assertRaises(ValueError):validate_model_view(bad)
        altered=copy.deepcopy(ep);altered['events'][-1]['entity']=2
        self.assertEqual(encode_numeric(ep)[:-1],encode_numeric(altered)[:-1])

    def test_all_arm_causality_reset_pairing_and_disabled_identity(self):
        Model,torch_module=model_classes()
        torch.manual_seed(17);x=torch.randn(2,6,8)
        ev=[{'kind':'SET','entity':0,'operator':None,'value':2},
            {'kind':'OP','entity':0,'operator':1,'value':None},
            {'kind':'QUERY','entity':0,'operator':None,'value':None},
            {'kind':'REVOKE','entity':0,'operator':None,'value':None},
            {'kind':'QUERY','entity':0,'operator':None,'value':None},
            {'kind':'NOOP','entity':0,'operator':None,'value':None}]
        events=[ev,copy.deepcopy(ev)];changed=x.clone();changed[:,4:]+=100
        m0=Model('M0',11)
        for arm in ('M0','M1','M2','M3','MA','MC'):
            m=Model(arm,11);y,_=m(x,events);y2,_=m(changed,events);again,_=m(x,events)
            torch.testing.assert_close(y[:,:4],y2[:,:4],rtol=0,atol=0)
            torch.testing.assert_close(y,again,rtol=0,atol=0)
            torch.testing.assert_close(m.head.weight,m0.head.weight,rtol=0,atol=0)
            torch.testing.assert_close(m.head.bias,m0.head.bias,rtol=0,atol=0)
        a=StructuredTransition(dtype=torch.float32);state=torch.randn(2,8)
        y,ns,_=a.step(x[:,0],state,control={'enabled':False})
        torch.testing.assert_close(y,x[:,0],rtol=0,atol=0);torch.testing.assert_close(ns,state,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
