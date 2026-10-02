"""Frozen, bounded component trials; all outputs remain in algebra/."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import time
import unittest
import numpy as np
from scipy.stats import t as student_t
import torch
from operators import (s5_matrices, StructuredTransition, scan, compose,
                       recompress_identity)
import test_algebra
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
DT=torch.float64


def metrics(actual, expected):
    diff=(actual-expected).detach()
    denom=expected.detach().norm().item()
    return {'relative_norm_error':diff.norm().item()/denom if denom else None,
            'absolute_squared_error':diff.square().sum().item(),
            'relative_squared_error':diff.square().sum().item()/denom**2 if denom else None}


def quantiles(xs):
    a=np.asarray(xs,dtype=np.float64)
    return dict(zip(('p50','p95','p99','max'),map(float,np.quantile(a,[.5,.95,.99,1]))))


def closure_diagnostics():
    g=torch.Generator().manual_seed(7101)
    # Exact small S5 matrices, affine offsets and independent boundary state.
    ids=torch.randint(10,(24,),generator=g)
    a=s5_matrices()[ids].unsqueeze(1)
    b=torch.randn(24,1,5,generator=g,dtype=DT)*.1
    h=torch.randn(1,5,generator=g,dtype=DT)
    seq=scan(a,b,h)
    exact={mode:metrics(scan(a,b,h,mode,4),seq) for mode in ('chunk','tree')}
    # Independent held-out probes for a larger switched rank-4 product.
    m=StructuredTransition(64,4,10,seed=53)
    with torch.no_grad():
        m.alpha_raw.fill_(4.)
        ops=m.matrices(ids)
        eye=torch.eye(64,dtype=DT)
        exact_product=eye
        for op in ops: exact_product=op@exact_product
        probes=torch.randn(32,64,generator=torch.Generator().manual_seed(8101),dtype=DT)
        direct=probes
        for op in ops: direct=(op@direct.T).T
        def approx_fold(items):
            p=eye
            for op in items: p=recompress_identity(op@p,4)
            return p
        def approx_tree(items):
            if len(items)==1: return recompress_identity(items[0],4)
            mid=len(items)//2
            return recompress_identity(approx_tree(items[mid:])@approx_tree(items[:mid]),4)
        compressed=approx_fold(ops)
        chunked=approx_fold([approx_fold(ops[i:i+4]) for i in range(0,len(ops),4)])
        tree=approx_tree(ops)
        out={name:metrics((mat@probes.T).T,direct) for name,mat in
             [('uncompressed_dense',exact_product),('recompressed_sequential',compressed),
              ('recompressed_chunk',chunked),('recompressed_tree',tree)]}
        out['tree_association']=metrics((tree@probes.T).T,(compressed@probes.T).T)
        # Homogeneous contraction checks and perturbation amplification along products.
        product=eye;growth=[];norms=[]
        initial=torch.randn(64,generator=g,dtype=DT);state=initial.clone();states=[]
        for op in ops:
            product=op@product;state=op@state
            growth.append(torch.linalg.matrix_norm(product,2).item())
            norms.append(torch.linalg.matrix_norm(op,2).item());states.append(state.norm().item())
        homogeneous={'max_local_operator_norm':max(norms),'product_growth':quantiles(growth),
                     'state_norms':quantiles(states),'common_contraction_bound':.99**24,
                     'final_perturbation_gain':torch.linalg.matrix_norm(product,2).item(),
                     'spectral_radius_alone_certifies_switching':False}
        shear=torch.tensor([[.9,1.],[0.,.9]],dtype=DT)
        negative={'local_spectral_radius':float(torch.linalg.eigvals(shear).abs().max()),
                  'switched_24step_product_norm':float(torch.linalg.matrix_norm(torch.linalg.matrix_power(shear.T@shear,12),2))}
    return {'small_exact_fp64':exact,'large_dim':64,'held_out_random_probes':32,
            'large_probe_errors':out,'homogeneous_stability':homogeneous,
            'switching_negative':negative,'compression_is_approximate':True}


def fit_cell(seed, rank, commuting, protocol):
    start=time.perf_counter()
    m=StructuredTransition(5,rank,10,commuting=commuting,seed=seed)
    ids=torch.arange(10)
    target=s5_matrices()
    fit=torch.randn(10,32,5,generator=torch.Generator().manual_seed(seed+1000),dtype=DT)
    evaluation=torch.randn(10,64,5,generator=torch.Generator().manual_seed(seed+2000),dtype=DT)
    optimizer=torch.optim.Adam(m.parameters(),lr=protocol['learning_rate'])
    grads=[];losses=[];updates=0
    for update in range(protocol['updates']):
        if time.perf_counter()-start>protocol['job_wall_seconds']:
            return dict(seed=seed,rank=rank,commuting=commuting,status='censored_timeout',updates=updates,losses=losses)
        optimizer.zero_grad()
        predicted=torch.einsum('kij,kpj->kpi',m.matrices(ids),fit)
        expected=torch.einsum('kij,kpj->kpi',target,fit)
        loss=(predicted-expected).square().mean()
        if not torch.isfinite(loss):
            return dict(seed=seed,rank=rank,commuting=commuting,status='failed_nonfinite',updates=updates,losses=losses)
        loss.backward()
        gn=torch.sqrt(sum(p.grad.square().sum() for p in m.parameters()))
        if not torch.isfinite(gn):
            return dict(seed=seed,rank=rank,commuting=commuting,status='failed_nonfinite_gradient',updates=updates,losses=losses)
        grads.append(gn.item());losses.append(loss.item());optimizer.step();updates+=1
    with torch.no_grad():
        matrices=m.matrices(ids)
        pred=torch.einsum('kij,kpj->kpi',matrices,evaluation)
        expected=torch.einsum('kij,kpj->kpi',target,evaluation)
        probe_errors=metrics(pred,expected)
        # Argmax action accuracy counts 50 independent permutation columns.
        point_accuracy=(matrices.argmax(-2)==target.argmax(-2)).double().mean().item()
        tokens=torch.randint(10,(64,8),generator=torch.Generator().manual_seed(seed+3000))
        state=torch.eye(5,dtype=DT).expand(64,-1,-1).clone()
        oracle=state.clone();tails=[]
        for k in range(8):
            state=matrices[tokens[:,k]]@state;oracle=target[tokens[:,k]]@oracle
            tails.extend(state.norm(dim=(-2,-1)).tolist())
        sequence_accuracy=(state.argmax(-2)==oracle.argmax(-2)).all(-1).double().mean().item()
        commutators=[(matrices[i]@matrices[j]-matrices[j]@matrices[i]).norm().item()
                     for i in range(10) for j in range(i)]
        budget=m.stats(torch.zeros(64,5,dtype=DT),10)
        latency_x=torch.zeros(1,5,dtype=DT);latency_h=torch.ones_like(latency_x)
        latency_control={'op_id':torch.tensor([0])}
        m.step(latency_x,latency_h,control=latency_control)
        latency_start=time.perf_counter()
        for _ in range(20): m.step(latency_x,latency_h,control=latency_control)
        latency=(time.perf_counter()-latency_start)/20
    return dict(seed=seed,rank=rank,commuting=commuting,status='executed',updates=updates,
                elapsed_seconds=time.perf_counter()-start,losses=losses,
                evaluation_probe_error=probe_errors,point_action_accuracy=point_accuracy,
                held_out_sequence_accuracy=sequence_accuracy,
                gradient_norms=quantiles(grads),state_norms=quantiles(tails),budget=budget,
                mean_step_latency_seconds_batch1=latency,latency_samples=20,
                max_commutator_norm=max(commutators),
                fit_probe_sha256=hashlib.sha256(fit.numpy().tobytes()).hexdigest(),
                eval_probe_sha256=hashlib.sha256(evaluation.numpy().tobytes()).hexdigest(),
                sequence_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    protocol_hash=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    start=time.perf_counter()
    with (HERE/'tests.log').open('w',encoding='utf-8') as log:
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_algebra)
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    results={'scope':protocol['scope'],'protocol_sha256':protocol_hash,
             'device':'CPU','torch_threads':torch.get_num_threads(),'tests':
             dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),
             'closure':closure_diagnostics(),'cells':[]}
    training_start=time.perf_counter()
    for rank in protocol['ranks']:
        for seed in protocol['paired_seeds']:
            for commuting in (False,True):
                if time.perf_counter()-training_start>protocol['aggregate_learned_wall_seconds']:
                    cell=dict(seed=seed,rank=rank,commuting=commuting,status='unattempted_aggregate_budget')
                elif not result.wasSuccessful():
                    cell=dict(seed=seed,rank=rank,commuting=commuting,status='unattempted_correctness_failure')
                else:
                    try: cell=fit_cell(seed,rank,commuting,protocol)
                    except Exception as exc: cell=dict(seed=seed,rank=rank,commuting=commuting,status='failed_exception',error=repr(exc))
                results['cells'].append(cell)
                (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
                print(f"rank={rank} seed={seed} commuting={commuting} {cell['status']}",flush=True)
    paired=[]
    for rank in protocol['ranks']:
        cells=[c for c in results['cells'] if c['rank']==rank]
        if all(c['status']=='executed' for c in cells):
            deltas=[next(c['held_out_sequence_accuracy'] for c in cells if c['seed']==s and not c['commuting'])-
                    next(c['held_out_sequence_accuracy'] for c in cells if c['seed']==s and c['commuting']) for s in protocol['paired_seeds']]
            mean=float(np.mean(deltas));half=float(student_t.ppf(.975,4)*np.std(deltas,ddof=1)/np.sqrt(5))
            paired.append(dict(rank=rank,sequence_accuracy_nc_minus_commuting=deltas,mean=mean,t95_df4=[mean-half,mean+half]))
        else: paired.append(dict(rank=rank,status='undefined_incomplete_pairs'))
    results['paired']=paired;results['elapsed_seconds']=time.perf_counter()-start
    results['learned_elapsed_seconds']=time.perf_counter()-training_start
    results['prepared_unattempted']=protocol['prepared_unattempted']
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'tests':results['tests'],'paired':paired,'elapsed_seconds':results['elapsed_seconds']}))


if __name__=='__main__':main()
