"""No-update positive/negative C6 checks. Optimizers and learned trials absent."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import sys
sys.dont_write_bytecode=True
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import unittest
import torch
from torch import nn
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from research.mamba3h.memory.slots import CausalSlotMemory,SparseRetrievalAttention,Control,OracleRoutes,route_oracle,COUNTERS
from research.mamba3h.memory.p1_checks.routing import Operation,causal_control

SEEDS=(11,23,37,53,71)
ARMS=('anonymous','symbolic_ADDRESS','sparse_anonymous_store_all')
FROZEN=Path(__file__).resolve().parents[1]


def fixture():
    """Three examples sharing 3 distinct writes and NOOPs; query each entity.

    D8 = four ID one-hot channels plus four visible value channels. QUERY/NOOP
    have no value channels. Targets are returned separately and never routed.
    """
    xs=torch.zeros(3,6,8,dtype=torch.float64)
    metadata=[]
    for b in range(3):
        ops=[]
        for t,(entity,value) in enumerate(((0,2),(1,0),(2,3))):
            xs[b,t,entity]=1
            xs[b,t,4+value]=1
            ops.append(Operation('WRITE',entity))
        for t,entity in ((3,3),(4,1)):
            xs[b,t,entity]=1
            ops.append(Operation('NOOP',entity))
        xs[b,5,b]=1
        ops.append(Operation('QUERY',b))
        metadata.append(ops)
    return xs,metadata,torch.tensor([2,0,3],dtype=torch.long)


class Probe(nn.Module):
    def __init__(self,arm):
        super().__init__()
        cls=SparseRetrievalAttention if arm=='sparse_anonymous_store_all' else CausalSlotMemory
        self.memory=cls(8,6,2).double()
        self.readout=nn.Linear(8,4).double()
        self.arm=arm

    def forward(self,x,operations,*,literal=False):
        state=self.memory.initial_state(x.shape[0])
        trace=[]
        for t in range(6):
            current=[ops[t] for ops in operations]
            symbolic=self.arm=='symbolic_ADDRESS'
            mode='store_all' if self.arm=='sparse_anonymous_store_all' else ('WRA' if literal else 'WR')
            control=causal_control(mode,current,state,symbolic_address=symbolic,literal_address=literal)
            y,state,stats=self.memory.step(x[:,t],state,control)
            trace.append({'stats':stats,'state_clock':state.clock.tolist(),
                          'keys':state.keys.detach().clone(),'values':state.values.detach().clone(),
                          'timestamps':state.timestamps.clone(),'occupied':state.occupied.clone(),
                          'y':y.detach().clone()})
        return self.readout(y),state,trace


def fingerprint(model):
    return hashlib.sha256(b''.join(p.detach().contiguous().numpy().tobytes() for p in model.parameters())).hexdigest()


def matched(seed):
    torch.manual_seed(seed)
    original=Probe('anonymous')
    result={arm:Probe(arm) for arm in ARMS}
    for model in result.values():model.load_state_dict(original.state_dict())
    return result


def gradients(model):
    return {name:{'nonzero_elements':0 if p.grad is None else int(torch.count_nonzero(p.grad)),
                  'norm':0. if p.grad is None else float(p.grad.norm()),
                  'max_abs':0. if p.grad is None else float(p.grad.abs().max()),
                  'finite':p.grad is None or bool(torch.isfinite(p.grad).all())}
            for name,p in model.named_parameters()}


def run_comparison():
    x,metadata,targets=fixture()
    rows=[]
    for seed in SEEDS:
        models=matched(seed)
        for arm,model in models.items():
            before=fingerprint(model)
            logits,state,trace=model(x,metadata)
            loss=nn.functional.cross_entropy(logits,targets)
            loss.backward()
            grads=gradients(model)
            stats=trace[-1]['stats']
            expected={'anonymous':3,'symbolic_ADDRESS':1,'sparse_anonymous_store_all':5}[arm]
            assert stats['similarity_evaluations']==x.shape[0]*expected
            assert all(row['stats']['evictions']==0 for row in trace)
            assert before==fingerprint(model),'no parameter update is authorized'
            assert all(g['finite'] for g in grads.values())
            q,k=grads['memory.query.weight'],grads['memory.key.weight']
            if arm=='symbolic_ADDRESS':
                assert q['nonzero_elements']==k['nonzero_elements']==0
            else:
                assert q['nonzero_elements']>0 and k['nonzero_elements']>0
            rows.append({'seed':seed,'arm':arm,'initial_model_sha256':before,'final_model_sha256':fingerprint(model),
                         'updates':0,'loss_at_initialization':float(loss.detach()),
                         'query_candidate_count_per_sample':expected,
                         'query_stats':stats,'total_operations':{name:int(state.counters[:,i].sum()) for i,name in enumerate(COUNTERS)},
                         'parameter_count':sum(p.numel() for p in model.parameters()),
                         'parameter_bytes_including_readout':sum(p.numel()*p.element_size() for p in model.parameters()),
                         'state_bytes_batch3':state.tensor_bytes(),'state_bytes_per_sample':state.tensor_bytes()//3,
                         'gradients':grads,'qk_nonzero_elements':q['nonzero_elements']+k['nonzero_elements'],
                         'dtype':str(x.dtype),'occupancy_by_step':[r['stats']['occupancy'] for r in trace],
                         'future_current_query_slot_in_selected':False})
    return rows


class PositiveAndCausalTests(unittest.TestCase):
    def test_frozen_original_hashes(self):
        import json
        manifest=json.loads((FROZEN/'manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((FROZEN/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(hashlib.sha256((FROZEN/'manifest.json').read_bytes()).hexdigest(),
                         '376ba6fb332a77235950c9a3ad0fb41daa2f488b60c7360d60e76848417b6149')

    def test_fixture_validity(self):
        x,metadata,targets=fixture()
        self.assertEqual(tuple(x.shape),(3,6,8))
        self.assertEqual(set(targets.tolist()),{0,2,3})
        for b,ops in enumerate(metadata):
            self.assertEqual([op.kind for op in ops],['WRITE']*3+['NOOP']*2+['QUERY'])
            self.assertEqual(len({op.entity for op in ops[:3]}),3)
            self.assertIn(ops[-1].entity,{op.entity for op in ops[:3]})
            self.assertEqual(float(x[b,3:,4:].abs().sum()),0.)

    def test_shared_initial_weights_and_no_forced_eviction(self):
        x,metadata,_=fixture()
        for seed in SEEDS:
            models=matched(seed)
            self.assertEqual(len({fingerprint(m) for m in models.values()}),1)
            states=[]
            for arm,model in models.items():
                _,state,trace=model(x,metadata)
                self.assertEqual(sum(r['stats']['evictions'] for r in trace),0)
                self.assertEqual(trace[4]['stats']['occupancy'],[5]*3 if arm=='sparse_anonymous_store_all' else [3]*3)
                self.assertEqual(trace[5]['stats']['occupancy'],[6]*3 if arm=='sparse_anonymous_store_all' else [3]*3)
                for t,row in enumerate(trace):
                    stored=t+1 if arm=='sparse_anonymous_store_all' else min(t+1,3)
                    for slot in range(stored):
                        torch.testing.assert_close(row['keys'][:,slot],model.memory.key(x[:,slot]))
                        torch.testing.assert_close(row['values'][:,slot],model.memory.value(x[:,slot]))
                states.append(state.tensor_bytes())
            self.assertEqual(len(set(states)),1)

    def test_candidate_counts_and_positive_qk_gradients(self):
        rows=run_comparison()
        self.assertEqual(len(rows),15)
        self.assertTrue(all(r['updates']==0 for r in rows))

    def test_all_arm_outputs_against_independent_prefix_attention(self):
        x,metadata,_=fixture()
        for seed in SEEDS:
            for arm,model in matched(seed).items():
                actual,_,_=model(x,metadata)
                expected=[]
                for b in range(3):
                    ids=[b] if arm=='symbolic_ADDRESS' else list(range(5 if arm=='sparse_anonymous_store_all' else 3))
                    keys=model.memory.key(x[b,ids])
                    scores=keys@model.memory.query(x[b,5])/math.sqrt(8)
                    order=sorted(range(len(ids)),key=lambda i:(-float(scores[i].detach()),ids[i]))[:2]
                    selected=[ids[i] for i in order]
                    r=torch.softmax(scores[order],dim=0)@model.memory.value(x[b,selected])
                    expected.append(model.readout(x[b,5]+model.memory.output(r)))
                torch.testing.assert_close(actual,torch.stack(expected),rtol=1e-12,atol=1e-12)

    def test_positive_constructed_readout_decodes_visible_writes_all_arms(self):
        x,metadata,targets=fixture()
        models=matched(11)
        for model in models.values():
            with torch.no_grad():
                for layer in (model.memory.query,model.memory.key,model.memory.value,model.memory.output):
                    layer.weight.copy_(torch.eye(8,dtype=torch.float64))
                model.readout.weight.zero_()
                model.readout.weight[:,4:]=torch.eye(4,dtype=torch.float64)
                model.readout.bias.zero_()
        self.assertEqual(len({fingerprint(m) for m in models.values()}),1)
        for model in models.values():
            logits,_,_=model(x,metadata)
            self.assertEqual(logits.argmax(-1).tolist(),targets.tolist())
        # This is a constructed representational witness, never a trained score.

    def test_literal_WRA_is_redundant_with_symbolic_ADDRESS(self):
        x,metadata,_=fixture()
        model=matched(11)['symbolic_ADDRESS']
        implicit,state,trace=model(x,metadata)
        literal,lstate,ltrace=model(x,metadata,literal=True)
        torch.testing.assert_close(implicit,literal,rtol=0,atol=0)
        torch.testing.assert_close(state.keys,lstate.keys,rtol=0,atol=0)
        torch.testing.assert_close(state.values,lstate.values,rtol=0,atol=0)
        self.assertEqual(trace[-1]['stats']['similarity_evaluations'],3)
        self.assertEqual(ltrace[-1]['stats']['similarity_evaluations'],3)
        self.assertEqual(trace[-1]['stats']['topk_calls'],3)
        self.assertEqual(ltrace[-1]['stats']['topk_calls'],0)
        self.assertEqual(ltrace[-1]['stats']['address_candidates'],3)
        self.assertEqual(trace[-1]['stats']['selected_indices'],ltrace[-1]['stats']['selected_indices'])

    def test_literal_WRA_without_symbolic_filter_still_addresses_prefix(self):
        x,metadata,_=fixture()
        m=matched(23)['symbolic_ADDRESS'].memory
        state=m.initial_state(3)
        for t in range(5):
            c=causal_control('WR',[ops[t] for ops in metadata],state,symbolic_address=True)
            _,state,_=m.step(x[:,t],state,c)
        current=[ops[5] for ops in metadata]
        symbolic=causal_control('WR',current,state,symbolic_address=True)
        address=causal_control('WRA',current,state,symbolic_address=False,literal_address=True)
        self.assertEqual(address.entity.tolist(),[-1]*3)
        self.assertEqual(address.address.tolist(),[[0,-1],[1,-1],[2,-1]])
        ys,_,ss=m.step(x[:,5],state,symbolic)
        ya,_,sa=m.step(x[:,5],state,address)
        torch.testing.assert_close(ys,ya,rtol=0,atol=0)
        self.assertEqual(sa['similarity_evaluations'],ss['similarity_evaluations'])

    def test_W_WR_WRA_literal_semantics_and_no_answer_fields(self):
        m=matched(11)['symbolic_ADDRESS'].memory
        state=m.initial_state(1)
        for mode in ('W','WR','WRA'):
            c=causal_control(mode,[Operation('WRITE',2)],state,symbolic_address=True)
            self.assertTrue(c.write[0])
            self.assertFalse(c.read[0])
            q=causal_control(mode,[Operation('QUERY',2)],state,symbolic_address=True,literal_address=mode=='WRA')
            self.assertFalse(q.write[0])
            self.assertEqual(bool(q.read[0]),mode!='W')
        with self.assertRaises(TypeError):Operation('QUERY',1,answer=3)
        with self.assertRaises(TypeError):Operation('QUERY',1,value=3)
        with self.assertRaises(TypeError):causal_control('WR',[{'kind':'QUERY','entity':1,'target':3}],state)
        for mode in ('W','WR','store_all'):
            with self.assertRaises(ValueError):causal_control(mode,[Operation('QUERY',1)],state,literal_address=True)

    def test_prefix_content_is_model_derived_and_query_write_invisible(self):
        x,metadata,_=fixture()
        model=matched(71)['sparse_anonymous_store_all']
        _,state,trace=model(x,metadata)
        for t,row in enumerate(trace):
            for slot in range(t+1):
                torch.testing.assert_close(row['keys'][:,slot],model.memory.key(x[:,slot]))
                torch.testing.assert_close(row['values'][:,slot],model.memory.value(x[:,slot]))
                self.assertTrue((row['timestamps'][:,slot]==slot).all())
                self.assertLess(slot,row['state_clock'][0])
        for selected in trace[-1]['stats']['selected_indices']:
            self.assertNotIn(5,selected)

    def test_future_contents_change_cannot_change_past_query(self):
        x,metadata,_=fixture()
        model=matched(37)['anonymous']
        state=model.memory.initial_state(3)
        for t in range(5):
            c=causal_control('WR',[ops[t] for ops in metadata],state)
            _,state,_=model.memory.step(x[:,t],state,c)
        # The unused slot has no causal evidence. Its value cannot participate.
        clean,_,stats=model.memory.step(x[:,5],state,causal_control('WR',[ops[5] for ops in metadata],state))
        future_contents=replace(state,keys=state.keys.clone(),values=state.values.clone())
        future_contents.keys[:,5]=1e5
        future_contents.values[:,5]=-1e5
        altered,_,st=model.memory.step(x[:,5],future_contents,causal_control('WR',[ops[5] for ops in metadata],future_contents))
        torch.testing.assert_close(clean,altered,rtol=0,atol=0)
        self.assertEqual(stats['selected_indices'],st['selected_indices'])
        for value in (-999.,999.):
            model.memory.step(torch.full((3,8),value,dtype=torch.float64),state,Control(torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool)))
        again,_,_=model.memory.step(x[:,5],state,causal_control('WR',[ops[5] for ops in metadata],state))
        torch.testing.assert_close(clean,again,rtol=0,atol=0)

    def test_future_slot_and_literal_address_rejected(self):
        x,metadata,_=fixture()
        model=matched(53)['symbolic_ADDRESS']
        _,state,_=model(x,metadata)
        future=replace(state,timestamps=state.timestamps.clone())
        future.timestamps[:,0]=state.clock
        with self.assertRaises(ValueError):causal_control('WRA',[Operation('QUERY',0)]*3,future,literal_address=True)
        with self.assertRaises(ValueError):model.memory.step(x[:,5],future,Control(torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool)))
        address=Control(torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),
                        address=torch.tensor([[5,-1]]*3))
        with self.assertRaises(ValueError):model.memory.step(x[:,5],state,address)


if __name__=='__main__':unittest.main(verbosity=2)
