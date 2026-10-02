"""CPU correctness and adversarial tests; run python -m unittest ...test_slots."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import unittest
from dataclasses import replace
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from .slots import CausalSlotMemory, SparseRetrievalAttention, Control, OracleRoutes, route_oracle, COUNTERS


def ctl(write=False,read=False,entity=-1,revoke=-1,batch=1,**kw):
    return Control(torch.full((batch,),write,dtype=torch.bool),
                   torch.full((batch,),read,dtype=torch.bool),
                   torch.full((batch,),entity,dtype=torch.long),
                   torch.full((batch,),revoke,dtype=torch.long),**kw)


def identity(capacity=3,top_k=1,batch=1,cls=CausalSlotMemory):
    m=cls(2,capacity,top_k).double()
    with torch.no_grad():
        for p in (m.query.weight,m.key.weight,m.value.weight,m.output.weight):
            p.copy_(torch.eye(2,dtype=torch.float64))
    return m,m.initial_state(batch)


class SlotTests(unittest.TestCase):
    def test_pre_read_and_current_write_not_visible(self):
        m,s=identity()
        x=torch.tensor([[1.,2.]],dtype=torch.float64)
        y,s,stats=m.step(x,s,ctl(True,True))
        torch.testing.assert_close(y,x)
        torch.testing.assert_close(s.keys[0,0],x[0])
        torch.testing.assert_close(s.values[0,0],x[0])
        self.assertEqual(stats['similarity_evaluations'],0)
        query=torch.tensor([[2.,0.]],dtype=torch.float64)
        y,s,stats=m.step(query,s,ctl(True,True))
        torch.testing.assert_close(y,query+x)
        # New K/V use query, not the post-retrieval y.
        torch.testing.assert_close(s.values[0,1],query[0])

    def test_exact_topk_and_ties(self):
        m,s=identity(4,2)
        for row in ([1.,0.],[3.,0.],[3.,1.],[2.,0.]):
            _,s,_=m.step(torch.tensor([row],dtype=torch.float64),s,ctl(True))
        x=torch.tensor([[1.,0.]],dtype=torch.float64)
        y,_,stats=m.step(x,s,ctl(read=True))
        self.assertEqual(stats['selected_indices'],[[1,2]])
        torch.testing.assert_close(y,x+torch.tensor([[3.,.5]],dtype=torch.float64))
        self.assertEqual(stats['similarity_evaluations'],4)
        self.assertEqual(stats['similarity_madds'],8)
        self.assertEqual(stats['value_reads'],2)

    def test_no_read_avoids_search(self):
        m,s=identity()
        _,s,_=m.step(torch.ones(1,2,dtype=torch.float64),s,ctl(True))
        # Fail if either actual sorting or softmax is called.
        from unittest.mock import patch
        with patch('torch.argsort',side_effect=AssertionError('search')),patch('torch.softmax',side_effect=AssertionError('read')):
            x=torch.zeros(1,2,dtype=torch.float64)
            y,_,stats=m.step(x,s,ctl())
        torch.testing.assert_close(x,y)
        for key in ('reads','similarity_evaluations','similarity_madds','topk_calls','topk_candidates','value_reads'):
            self.assertEqual(stats[key],0)

    def test_overwrite_removes_stale_evidence_before_read(self):
        m,s=identity()
        _,s,_=m.step(torch.tensor([[1.,4.]],dtype=torch.float64),s,ctl(True,entity=7))
        x=torch.tensor([[2.,3.]],dtype=torch.float64)
        y,s,stats=m.step(x,s,ctl(True,True,entity=7))
        torch.testing.assert_close(y,x)
        self.assertEqual(stats['overwrites'],1)
        self.assertEqual(s.occupied.sum(),1)
        y,_,_=m.step(torch.zeros_like(x),s,ctl(read=True,entity=7))
        torch.testing.assert_close(y,x)

    def test_revoke_clears_contents_and_read(self):
        m,s=identity()
        _,s,_=m.step(torch.ones(1,2,dtype=torch.float64),s,ctl(True,entity=3))
        y,s,stats=m.step(torch.zeros(1,2,dtype=torch.float64),s,ctl(read=True,entity=3,revoke=3))
        self.assertEqual(stats['revocations'],1)
        self.assertEqual(stats['similarity_evaluations'],0)
        self.assertFalse(s.occupied.any())
        self.assertEqual(s.keys.abs().sum(),0)
        self.assertEqual(s.values.abs().sum(),0)
        self.assertEqual(y.abs().sum(),0)

    def test_entity_and_batch_isolation(self):
        m,s=identity(batch=2)
        x=torch.tensor([[1.,2.],[9.,7.]],dtype=torch.float64)
        _,s,_=m.step(x,s,ctl(True,entity=4,batch=2))
        y,_,_=m.step(torch.zeros_like(x),s,ctl(read=True,entity=4,batch=2))
        torch.testing.assert_close(y,x)
        y,_,stats=m.step(torch.zeros_like(x),s,ctl(read=True,entity=5,batch=2))
        self.assertEqual(y.abs().sum(),0)
        self.assertEqual(stats['similarity_evaluations'],0)

    def test_eviction_and_fixed_budget_store_all(self):
        m,s=identity(2)
        bytes0=s.tensor_bytes()
        for i in range(6):
            base=ctl()
            _,s,stats=m.step(torch.tensor([[float(i),1.]],dtype=torch.float64),s,route_oracle('store_all',base))
            self.assertLessEqual(s.occupied.sum(),2)
            self.assertEqual(s.tensor_bytes(),bytes0)
            self.assertEqual(stats['evictions'],int(i>=2))
        self.assertEqual(sorted(s.values[0,:,0].tolist()),[4.,5.])
        self.assertEqual(s.counters[0,COUNTERS.index('writes')],6)
        self.assertEqual(s.counters[0,COUNTERS.index('evictions')],4)

    def test_retention_drops_clear(self):
        m,s=identity()
        _,s,_=m.step(torch.ones(1,2,dtype=torch.float64),s,ctl(True))
        _,s,stats=m.step(torch.zeros(1,2,dtype=torch.float64),s,ctl(read=True,retain=torch.zeros(1,3,dtype=torch.bool)))
        self.assertEqual(stats['retention_drops'],1)
        self.assertEqual(stats['similarity_evaluations'],0)
        self.assertEqual(s.values.abs().sum(),0)

    def test_oracle_only_routing_and_content_same(self):
        m,s=identity()
        x=torch.tensor([[1.,2.]],dtype=torch.float64)
        params={n:p.clone() for n,p in m.named_parameters()}
        reference=m.step(x,s,ctl(True))[1]
        for mode in ('W','WR','WRA'):
            routed=route_oracle(mode,ctl(),OracleRoutes(write=torch.tensor([True])))
            state=m.step(x,s,routed)[1]
            torch.testing.assert_close(state.keys,reference.keys)
            torch.testing.assert_close(state.values,reference.values)
        for n,p in m.named_parameters():
            torch.testing.assert_close(p,params[n])
        with self.assertRaises(TypeError):
            OracleRoutes(answer=torch.ones(1))
        with self.assertRaises(TypeError):
            Control(torch.tensor([True]),torch.tensor([False]),value=torch.ones(1))
        with self.assertRaises(TypeError):
            m.step(x,s,{'write':True,'target':1})
        with self.assertRaises(ValueError):
            route_oracle('W',ctl(),OracleRoutes(read=torch.tensor([True])))
        with self.assertRaises(ValueError):
            route_oracle('WR',ctl(),OracleRoutes(address=torch.tensor([[0]])))

    def test_address_uses_same_model_scores_softmax_and_values(self):
        m,s=identity(3,2)
        for row in ([1.,0.],[2.,1.],[3.,2.]):
            _,s,_=m.step(torch.tensor([row],dtype=torch.float64),s,ctl(True))
        x=torch.tensor([[1.,0.]],dtype=torch.float64)
        expected=x+torch.softmax(torch.tensor([1.,3.],dtype=torch.float64)/(2**.5),0) @ s.values[0,[0,2]]
        control=route_oracle('WRA',ctl(),OracleRoutes(read=torch.tensor([True]),address=torch.tensor([[0,2]])))
        y,_,stats=m.step(x,s,control)
        torch.testing.assert_close(y,expected)
        self.assertEqual(stats['topk_calls'],0)
        self.assertEqual(stats['similarity_evaluations'],2)
        self.assertEqual(stats['address_candidates'],2)

    def test_negative_future_stale_duplicate_cross_entity_address(self):
        m,s=identity(3,2)
        _,s,_=m.step(torch.ones(1,2,dtype=torch.float64),s,ctl(True,entity=1))
        x=torch.zeros(1,2,dtype=torch.float64)
        for addresses,entity in (([0,0],1),([1,-1],1),([0,-1],2),([3,-1],1),([-2,-1],1)):
            with self.subTest(addresses=addresses,entity=entity),self.assertRaises(ValueError):
                m.step(x,s,ctl(read=True,entity=entity,address=torch.tensor([addresses])))
        future=replace(s,timestamps=s.timestamps.masked_fill(s.occupied,int(s.clock[0])))
        with self.assertRaises(ValueError):
            m.step(x,future,ctl(read=True))

    def test_future_suffix_does_not_change_prefix(self):
        m,s=identity()
        prefix=[torch.tensor([[1.,2.]],dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64)]
        results=[]
        for suffix in (torch.tensor([[999.,0.]],dtype=torch.float64),torch.tensor([[-999.,4.]],dtype=torch.float64)):
            local=s
            _,local,_=m.step(prefix[0],local,ctl(True))
            y,local,_=m.step(prefix[1],local,ctl(read=True))
            m.step(suffix,local,ctl(True))
            results.append(y)
        torch.testing.assert_close(*results)

    def test_reset_and_input_state_immutable(self):
        m,s=identity()
        snapshots={f:getattr(s,f).clone() for f in s.__dataclass_fields__}
        _,s1,_=m.step(torch.ones(1,2,dtype=torch.float64),s,ctl(True))
        for f,snapshot in snapshots.items():
            torch.testing.assert_close(getattr(s,f),snapshot)
        _,fresh,stats=m.step(torch.zeros(1,2,dtype=torch.float64),m.initial_state(1),ctl(read=True))
        self.assertEqual(stats['value_reads'],0)
        self.assertFalse(fresh.occupied.any())
        self.assertTrue(s1.occupied.any())

    def test_gradients_reach_write_x_query_and_parameters(self):
        torch.manual_seed(11)
        m=CausalSlotMemory(2,3,2).double()
        s=m.initial_state(1)
        writes=[]
        for row in ([1.,2.],[2.,-1.]):
            x=torch.tensor([row],dtype=torch.float64,requires_grad=True)
            writes.append(x)
            _,s,_=m.step(x,s,ctl(True))
        q=torch.tensor([[.5,1.]],dtype=torch.float64,requires_grad=True)
        y,_,_=m.step(q,s,ctl(read=True))
        y.square().sum().backward()
        for x in writes+[q]:
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum(),0)
        for layer in (m.query,m.key,m.value,m.output):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum(),0)

    def test_sparse_attention_matched_projection_reference(self):
        m,s=identity(3,2)
        a,sa=identity(3,2,cls=SparseRetrievalAttention)
        a.load_state_dict(m.state_dict())
        for row,write,read in (([1.,2.],True,False),([2.,3.],True,False),([1.,0.],False,True)):
            x=torch.tensor([row],dtype=torch.float64)
            ym,s,ms=m.step(x,s,ctl(write,read))
            ya,sa,ats=a.step(x,sa,ctl(write,read))
            torch.testing.assert_close(ym,ya)
            self.assertEqual(ms['similarity_evaluations'],ats['similarity_evaluations'])
            self.assertEqual(ms['state_bytes'],ats['state_bytes'])

    def test_invalid_budget_shapes_and_nonfinite(self):
        for args in ((2,0,1),(2,2,3),(0,2,1)):
            with self.assertRaises(ValueError): CausalSlotMemory(*args)
        m,s=identity()
        for x in (torch.tensor([[float('nan'),0.]],dtype=torch.float64),torch.zeros(1,3,dtype=torch.float64)):
            with self.assertRaises(ValueError): m.step(x,s,ctl())
        with self.assertRaises(ValueError):
            m.step(torch.zeros(1,2,dtype=torch.float64),s,Control(torch.ones(1),torch.zeros(1)))

    def test_random_exact_topk_against_exhaustive_reference(self):
        generator=torch.Generator().manual_seed(71)
        for top_k in (1,2,4):
            m,s=identity(4,top_k)
            rows=torch.randn(4,2,generator=generator,dtype=torch.float64)
            for row in rows:
                _,s,_=m.step(row[None],s,ctl(True))
            for _ in range(7):
                x=torch.randn(1,2,generator=generator,dtype=torch.float64)
                scores=(rows@x[0])/(2**.5)
                chosen=sorted(range(4),key=lambda i:(-float(scores[i]),i))[:top_k]
                y,_,stats=m.step(x,s,ctl(read=True))
                self.assertEqual(stats['selected_indices'],[chosen])
                torch.testing.assert_close(y,x+torch.softmax(scores[chosen],0)@rows[chosen])
                self.assertEqual(stats['attention_madds'],top_k*2)

    def test_read_off_mixed_batch_counters_and_grad(self):
        m,s=identity(batch=2)
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64,requires_grad=True)
        _,s,_=m.step(x,s,ctl(True,batch=2))
        query=torch.zeros(2,2,dtype=torch.float64,requires_grad=True)
        y,_,stats=m.step(query,s,Control(torch.zeros(2,dtype=torch.bool),torch.tensor([False,True])))
        torch.testing.assert_close(y[0],query[0])
        torch.testing.assert_close(y[1],x[1])
        self.assertEqual(stats['reads'],1)
        self.assertEqual(stats['similarity_evaluations'],1)
        self.assertEqual(stats['candidate_slot_checks'],3)
        self.assertEqual(stats['projection_madds'],24)
        self.assertEqual(stats['output_madds'],8)
        y.sum().backward()
        self.assertEqual(x.grad[0].abs().sum(),0)
        self.assertGreater(x.grad[1].abs().sum(),0)

    def test_evicted_entity_cannot_be_retrieved(self):
        m,s=identity(1)
        for entity in (2,3):
            _,s,_=m.step(torch.ones(1,2,dtype=torch.float64)*entity,s,ctl(True,entity=entity))
        y,_,stats=m.step(torch.zeros(1,2,dtype=torch.float64),s,ctl(read=True,entity=2))
        self.assertEqual(y.abs().sum(),0)
        self.assertEqual(stats['similarity_evaluations'],0)

    def test_value_perturbation_cannot_change_attention_selection(self):
        m,s=identity(3,2)
        for row in ([1.,0.],[2.,1.],[3.,2.]):
            _,s,_=m.step(torch.tensor([row],dtype=torch.float64),s,ctl(True))
        query=torch.tensor([[1.,0.]],dtype=torch.float64)
        y,_,stats=m.step(query,s,ctl(read=True))
        # Only Q/K decide selection/logits; content does not enter addressing.
        perturbed=replace(s,values=s.values*999)
        yp,_,sp=m.step(query,perturbed,ctl(read=True))
        self.assertEqual(stats['selected_indices'],sp['selected_indices'])
        torch.testing.assert_close(yp-query,999*(y-query))


if __name__=='__main__':
    unittest.main(verbosity=2)
