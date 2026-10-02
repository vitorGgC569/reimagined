"""Root2 pretraining gates for the fresh positive addressing protocol."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from positive_runner import classes,numeric,run
Model,torch=classes()

def fixture():
    return [{'kind':'WRITE','entity':0,'value':1,'operator':None},
            {'kind':'WRITE','entity':1,'value':2,'operator':None},
            {'kind':'WRITE','entity':2,'value':3,'operator':None},
            {'kind':'NOOP','entity':0,'value':None,'operator':None},
            {'kind':'NOOP','entity':3,'value':None,'operator':None},
            {'kind':'QUERY','entity':0,'value':None,'operator':None}]

class Positive(unittest.TestCase):
    def test_unprepared_run_rejected_before_jobs_or_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'unprepared'
            with self.assertRaises(FileNotFoundError):run(output)
            self.assertFalse(output.exists())

    def test_numeric_whitelist_targets_future_independence(self):
        ep={'events':fixture(),'targets':[-100]*5+[1],'metadata':{'split':'train'}}
        poisoned=copy.deepcopy(ep);poisoned['targets']=[3]*6;poisoned['metadata']={'answer':3,'future':'payload'}
        np.testing.assert_array_equal(numeric(ep),numeric(poisoned))
        poisoned['events'][-1]['entity']=2
        np.testing.assert_array_equal(numeric(ep)[:-1],numeric(poisoned)[:-1])
        self.assertTrue(np.all(numeric(ep)[3:,4:]==0))

    def test_parameters_reservations_and_bitwise_initialization(self):
        models={arm:Model(arm,11) for arm in ('M0','MC_capacity','M2_symbolic_address','M2_anonymous_k2','MA_anonymous_k2')}
        x=torch.tensor(numeric({'events':fixture()}))[None]
        for arm,m in models.items():
            _,st=m(x,[fixture()]);self.assertEqual(st['reserved_aux_bytes'],630)
            self.assertEqual(st['active_aux_bytes'],0 if arm in ('M0','MC_capacity') else 630)
            self.assertEqual(sum(p.numel() for p in m.parameters()),36 if arm=='M0' else 319)
            torch.testing.assert_close(m.head.weight,models['M0'].head.weight,rtol=0,atol=0)
        a=models['M2_symbolic_address'].memory.state_dict()
        for arm in ('M2_anonymous_k2','MA_anonymous_k2'):
            for name,p in models[arm].memory.state_dict().items():torch.testing.assert_close(a[name],p,rtol=0,atol=0)

    def test_anonymous_QK_gradients_symbolic_zero_and_no_eviction(self):
        for seed in (11,23,37,53,71):
            torch.manual_seed(seed+333);x=torch.randn(1,6,8)
            for arm,expected in [('M2_symbolic_address',1),('M2_anonymous_k2',3),('MA_anonymous_k2',5)]:
                m=Model(arm,seed);out,st=m(x,[fixture()]);out.square().sum().backward()
                self.assertEqual(st['counters']['similarity_evaluations'],expected);self.assertEqual(st['counters']['evictions'],0)
                for name in ('query','key'):
                    g=getattr(m.memory,name).weight.grad
                    if arm=='M2_symbolic_address':self.assertEqual(float(g.norm()),0)
                    else:self.assertGreater(float(g.norm()),0)

    def test_explicit_state_reset(self):
        m=Model('M2_anonymous_k2',37);x=torch.tensor(numeric({'events':fixture()}))[None]
        a,_=m(x,[fixture()]);b,_=m(x,[fixture()]);torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
