import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import copy
import unittest
import torch
from exact_entity import ExactEntityControl,run_events,PAIRS
from research.mamba3h.benchmarks.oracle import solve
from research.mamba3h.benchmarks.schema import event,UNDEFINED,IGNORE


def episode(events): return dict(task='inst',group='s5',events=events)
def reference(events): return [None if v==IGNORE else 5 if v==UNDEFINED else v for v in solve(episode(events))]


class ExactEntityTests(unittest.TestCase):
    def test_every_observed_point_and_current_operator(self):
        for point in range(5):
            for operator in range(10):
                evs=[event('SET',0,value=point),event('OP',0,operator=operator),event('QUERY',0)]
                for per_entity in (False,True):
                    pred,states,_=run_events(evs,ExactEntityControl(1,per_entity))
                    self.assertEqual(pred,reference(evs))
                    self.assertEqual(int(states[-1][0,0].argmax()),pred[-1])
                    self.assertLess((states[-1].norm()-1).abs().item(),2e-12)

    def test_chronological_order_negative(self):
        correct=[event('SET',value=0),event('OP',operator=0),event('OP',operator=4),event('QUERY')]
        reversed_order=[correct[0],correct[2],correct[1],correct[3]]
        actual=run_events(correct,ExactEntityControl(1))[0]
        self.assertEqual(actual,reference(correct))
        self.assertNotEqual(actual[-1],run_events(reversed_order,ExactEntityControl(1))[0][-1])

    def test_global_interference_vs_entity_isolation_negative(self):
        evs=[event('SET',0,value=0),event('SET',1,value=4),event('OP',0,operator=0),
             event('QUERY',0),event('QUERY',1),event('REVOKE',0),event('QUERY',1)]
        per=run_events(evs,ExactEntityControl(2,True))[0]
        glob=run_events(evs,ExactEntityControl(2,False))[0]
        self.assertEqual(per,reference(evs))
        self.assertNotEqual(glob,per)
        self.assertEqual(per[-1],4);self.assertEqual(glob[-1],5)

    def test_overwrite_revoke_reinitialization_negatives(self):
        evs=[event('QUERY'),event('OP',operator=0),event('QUERY'),event('SET',value=0),
             event('OP',operator=0),event('QUERY'),event('SET',value=4),event('QUERY'),
             event('REVOKE'),event('OP',operator=9),event('QUERY'),event('SET',value=2),event('QUERY')]
        pred,states,_=run_events(evs,ExactEntityControl(1))
        self.assertEqual(pred,reference(evs))
        self.assertEqual([p for p in pred if p is not None],[5,5,1,4,5,2])
        self.assertTrue(torch.equal(states[8],torch.zeros_like(states[8])))
        ignore_overwrite=[dict(ev) for ev in evs]
        ignore_overwrite[6]=event('NOOP')
        self.assertNotEqual(run_events(ignore_overwrite,ExactEntityControl(1))[0][7],pred[7])
        ignore_revoke=[dict(ev) for ev in evs];ignore_revoke[8]=event('NOOP')
        self.assertNotEqual(run_events(ignore_revoke,ExactEntityControl(1))[0][10],pred[10])

    def test_query_neutral_and_past_state_are_nonmutating(self):
        m=ExactEntityControl(3);state=m.initial_state(1)
        _,state,_=m.step([event('SET',2,value=3)],state)
        original=state.clone()
        for ev in (event('QUERY',2),event('NOOP'),event('DISTRACTOR',2,value=124)):
            _,next_state,_=m.step([ev],state)
            self.assertTrue(torch.equal(next_state,original));self.assertTrue(torch.equal(state,original))
        _,next_state,_=m.step([event('OP',2,operator=9)],state)
        self.assertTrue(torch.equal(state,original))
        self.assertTrue(torch.equal(next_state[:,:2],state[:,:2]))

    def test_labels_metadata_and_future_do_not_enter_model(self):
        evs=[event('SET',value=0),event('OP',operator=0),event('QUERY'),event('SET',value=4),event('QUERY')]
        ep=episode(evs);ep['targets']=reference(evs);ep['metadata']={'seed':11}
        pred=run_events(ep['events'],ExactEntityControl(1))[0]
        poisoned=copy.deepcopy(ep);poisoned['targets']=[999]*len(evs);poisoned['metadata']={'seed':71,'answer':999}
        poisoned['events'][3]=event('SET',value=2)
        poison_pred=run_events(poisoned['events'],ExactEntityControl(1))[0]
        self.assertEqual(pred[:3],poison_pred[:3]);self.assertNotEqual(pred[-1],poison_pred[-1])
        for key in ('targets','answer','future','logits','solver','metadata'):
            with self.assertRaises(ValueError):
                ExactEntityControl(1).step([dict(evs[2],**{key:0})])
        with self.assertRaises(ValueError): ExactEntityControl(1).step([event('QUERY',value=4)])
        with self.assertRaises(ValueError): ExactEntityControl(1).step([ep])

    def test_batch_reset_and_explicit_state_bytes(self):
        per=ExactEntityControl(3,True);glob=ExactEntityControl(3,False)
        logits,state,stats=per.step([event('SET',0,value=1),event('SET',2,value=4)])
        self.assertEqual(state.shape,(2,3,5));self.assertEqual(stats['persistent_state_bytes'],240)
        _,_,gs=glob.step([event('SET',0,value=1),event('SET',2,value=4)])
        self.assertEqual(gs['persistent_state_bytes'],80)
        self.assertEqual(stats['parameter_count'],0);self.assertEqual(gs['parameter_count'],0)
        together=per.step([event('QUERY',0),event('QUERY',2)],state)[0]
        alone=per.step([event('QUERY',2)],state[1:])[0]
        torch.testing.assert_close(together[1:],alone)
        reset=per.step([event('QUERY',0)])[0]
        self.assertEqual(int(reset.argmax()),5)
        for dtype in (torch.float32,torch.float64):
            evs=[event('SET',value=0),event('OP',operator=0),event('QUERY')]
            self.assertEqual(run_events(evs,ExactEntityControl(1,dtype=dtype))[0],reference(evs))

    def test_readout_is_fixed_linear_model_no_solver_and_gradient(self):
        model=ExactEntityControl(1)
        self.assertEqual(sum(p.numel() for p in model.parameters()),0)
        h=torch.eye(5,dtype=torch.float64,requires_grad=True)
        logits=model.readout(h)
        self.assertTrue(torch.equal(logits.argmax(-1),torch.arange(5)))
        self.assertEqual(int(model.readout(torch.zeros(1,5,dtype=torch.float64)).argmax()),5)
        logits[:,:5].sum().backward()
        torch.testing.assert_close(h.grad,torch.ones_like(h))

    def test_invalid_events_and_states(self):
        m=ExactEntityControl(3)
        for ev in (event('SET',value=5),event('OP',operator=10),event('QUERY',entity=-1),event('WRITE',value=1)):
            with self.assertRaises(ValueError): m.step([ev])
        with self.assertRaises(ValueError): m.step([])
        with self.assertRaises(ValueError): m.step([event('QUERY')],torch.zeros(1,1,5,dtype=torch.float64))
        with self.assertRaises(ValueError): m.step([event('QUERY')],torch.full((1,3,5),float('nan'),dtype=torch.float64))


if __name__=='__main__':unittest.main(verbosity=2)
