"""Bounded CPU differential, backward, exact and negative tests."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import itertools
import unittest
import torch
from operators import (S5_PAIRS, transition, s5_matrices, permutation_matrix,
                       scan, recompress_identity, StructuredTransition)
torch.set_num_threads(2)
DT = torch.float64


class AlgebraTests(unittest.TestCase):
    def test_all_s5_actions_independent_point_solver(self):
        ops = s5_matrices()
        self.assertEqual(S5_PAIRS, tuple(itertools.combinations(range(5),2)))
        for perm in itertools.permutations(range(5)):
            p = permutation_matrix(perm)
            for token,(i,j) in enumerate(S5_PAIRS):
                op = list(range(5)); op[i],op[j]=j,i
                expected = tuple(op[v] for v in perm)
                torch.testing.assert_close(ops[token]@p, permutation_matrix(expected), atol=2e-15, rtol=2e-15)
        for op in ops:
            torch.testing.assert_close(op@op,torch.eye(5,dtype=DT),atol=2e-15,rtol=2e-15)

    def test_noncommutativity_and_wrong_order_negative(self):
        a,b=s5_matrices()[0],s5_matrices()[4] # (0,1), (1,2)
        self.assertGreater((a@b-b@a).norm().item(),1.)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(a@b,b@a)

    def test_transition_dense_and_gradcheck(self):
        g=torch.Generator().manual_seed(23)
        h=torch.randn(2,5,generator=g,dtype=DT,requires_grad=True)
        u=torch.randn(2,4,5,generator=g,dtype=DT,requires_grad=True)
        al=torch.rand(2,4,generator=g,dtype=DT,requires_grad=True)
        d=torch.rand(2,5,generator=g,dtype=DT,requires_grad=True)
        eye=torch.eye(5,dtype=DT).expand(2,-1,-1)
        a=torch.diag_embed(d)
        for j in range(4):
            v=u[:,j]/u[:,j].norm(dim=-1,keepdim=True)
            a=(eye-al[:,j,None,None]*v[:,:,None]*v[:,None,:])@a
        torch.testing.assert_close(transition(h,u,al,d),(a@h.unsqueeze(-1)).squeeze(-1))
        self.assertTrue(torch.autograd.gradcheck(transition,(h,u,al,d),fast_mode=True))

    def test_scan_forward_vjp_independent_reverse(self):
        g=torch.Generator().manual_seed(37)
        for t in (0,1,7,17):
            a=(torch.randn(t,2,5,5,generator=g,dtype=DT)*.1).requires_grad_()
            b=torch.randn(t,2,5,generator=g,dtype=DT,requires_grad=True)
            h=torch.randn(2,5,generator=g,dtype=DT,requires_grad=True)
            seeds=torch.randn(t,2,5,generator=g,dtype=DT)
            final_seed=torch.randn_like(h)
            ref=scan(a,b,h)
            prev=torch.cat([h.unsqueeze(0),ref[:-1]]) if t else h.new_empty((0,2,5))
            ga=torch.zeros_like(a);gb=torch.zeros_like(b);adj=final_seed
            for k in reversed(range(t)):
                adj=adj+seeds[k]
                ga[k]=adj[:,:,None]*prev[k,:,None,:]
                gb[k]=adj
                adj=(a[k].transpose(-1,-2)@adj.unsqueeze(-1)).squeeze(-1)
            for method in ('sequential','chunk','tree'):
                for chunk in (1,4,8):
                    out=scan(a,b,h,method,chunk)
                    torch.testing.assert_close(out,ref,atol=1e-12,rtol=1e-12)
                    loss=(out*seeds).sum()+((out[-1] if t else h)*final_seed).sum()
                    got=torch.autograd.grad(loss,(a,b,h),allow_unused=True)
                    for actual,expected in zip(got,(ga,gb,adj)):
                        if actual is None: actual=torch.zeros_like(expected)
                        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)

    def test_modules_parameter_matching_commuting_and_gradients(self):
        for rank in (1,2,4):
            nc=StructuredTransition(rank=rank)
            cc=StructuredTransition(rank=rank,commuting=True)
            self.assertEqual(sum(p.numel() for p in nc.parameters()),sum(p.numel() for p in cc.parameters()))
            self.assertEqual(sum(p.numel() for p in nc.parameters()),10*(rank*8+rank+8))
            a,b=cc.matrices(torch.tensor([0,1])).unbind()
            torch.testing.assert_close(a@b,b@a,atol=2e-14,rtol=2e-14)
            for model in (nc,cc):
                x=torch.randn(2,8,dtype=DT,requires_grad=True)
                h=torch.randn(2,8,dtype=DT,requires_grad=True)
                y,_,stats=model.step(x,h,control={'op_id':torch.tensor([0,1])})
                grads=torch.autograd.grad(y.square().sum(),(x,h,*model.parameters()))
                self.assertTrue(all(torch.isfinite(v).all() for v in grads))
                self.assertTrue(all(v.abs().sum()>0 for v in grads))
                self.assertEqual(stats['state_bytes'],128)
                # Directional parameter VJP independent centered finite difference.
                p=model.u; direction=torch.randn_like(p); direction/=direction.norm()
                pred=(grads[2]*direction).sum().item(); original=p.detach().clone(); eps=1e-6
                vals=[]
                for sign in (1,-1):
                    with torch.no_grad(): p.copy_(original+sign*eps*direction)
                    vals.append(model.step(x,h,control={'op_id':torch.tensor([0,1])})[0].square().sum().item())
                with torch.no_grad(): p.copy_(original)
                self.assertAlmostEqual(pred,(vals[0]-vals[1])/(2*eps),places=6)

    def test_recompression_not_closed_negative(self):
        a,b=s5_matrices()[0],s5_matrices()[7] # disjoint swaps, rank residual 2
        self.assertEqual(int(torch.linalg.matrix_rank(a@b-torch.eye(5,dtype=DT))),2)
        self.assertGreater((recompress_identity(a@b,1)-a@b).norm().item(),1.)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(recompress_identity(a@b,1),a@b)

    def test_affine_chunk_boundary_and_common_contraction(self):
        g=torch.Generator().manual_seed(71)
        m=StructuredTransition(8,4,10,seed=71)
        a=m.matrices(torch.randint(10,(11,),generator=g)).unsqueeze(1)
        b=torch.randn(11,1,8,generator=g,dtype=DT)
        h=torch.randn(1,8,generator=g,dtype=DT)
        whole=scan(a,b,h)
        left=scan(a[:4],b[:4],h)
        right=scan(a[4:],b[4:],left[-1])
        torch.testing.assert_close(torch.cat([left,right]),whole)
        wrong=scan(a[4:],b[4:],torch.zeros_like(h))
        self.assertGreater((wrong-right).norm().item(),.01)
        product=torch.eye(8,dtype=DT)
        for op in a[:,0]:
            self.assertLessEqual(torch.linalg.matrix_norm(op,2).item(),1.+1e-12)
            product=op@product
            self.assertLessEqual(torch.linalg.matrix_norm(product,2).item(),1.+1e-12)

    def test_switching_spectral_radius_not_certificate_negative(self):
        a=torch.tensor([[.9,1.],[0.,.9]],dtype=DT)
        b=a.T
        self.assertLess(torch.linalg.eigvals(a).abs().max().item(),1.)
        self.assertGreater(torch.linalg.matrix_power(b@a,12).norm().item(),100.)

    def test_isolation_disabled_and_invalid_controls(self):
        m=StructuredTransition()
        x=torch.randn(2,8,dtype=DT); h=torch.randn_like(x); original=h.clone()
        y,s,_=m.step(x,h,control={'enabled':False})
        self.assertTrue(torch.equal(y,x));self.assertTrue(torch.equal(s,h))
        m.step(x,h,control={'op_id':torch.tensor([0,1])})
        self.assertTrue(torch.equal(h,original))
        together=m.step(x,None,control={'op_id':torch.tensor([0,1])})[0]
        alone=m.step(x[1:],None,control={'op_id':torch.tensor([1])})[0]
        torch.testing.assert_close(together[1:],alone)
        for key in ('target','future','logits','answer'):
            with self.assertRaises(ValueError): m.step(x,h,control={'op_id':torch.tensor([0,1]),key:0})
        with self.assertRaises(ValueError): m.step(x,h,control={'op_id':torch.tensor([-1,10])})
        with self.assertRaises(ValueError): transition(x,torch.zeros(2,1,8,dtype=DT),torch.ones(2,1,dtype=DT))
        with self.assertRaises(ValueError): permutation_matrix([0,0,1])
        with self.assertRaises(ValueError): scan(torch.empty(0,2,8,8),torch.empty(0,2,8),x,'invalid')


if __name__=='__main__': unittest.main(verbosity=2)
