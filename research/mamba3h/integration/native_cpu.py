"""Actual frozen native Mamba3 CPU; separate process avoids OpenMP conflicts.

No PyTorch import, no GPU, no production writes. Feature mode freezes native
weights: learned pilot evidence concerns adapters, NOT end-to-end native training.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BUILD=ROOT/'OXN/nsos/build-gm-cpu'
FREEZE=ROOT/'research/mamba3h/manifests/freeze.json'
if FREEZE.exists():
    expected=next(e['sha256'] for e in json.loads(FREEZE.read_text())['files'] if e['path'].endswith('nsos_ext.cp312-win_amd64.pyd'))
    assert hashlib.sha256((BUILD/'nsos_ext.cp312-win_amd64.pyd').read_bytes()).hexdigest()==expected,'Frozen CPU provider drift; refusing experiment'
sys.path.insert(0,str(BUILD))
import nsos_ext as ns

FIELDS=('phase','ssm','k','v')
def nt(x): return ns.Tensor.from_numpy(np.ascontiguousarray(x,dtype=np.float32))
def state_arrays(state): return {k:getattr(state,k).numpy().copy() for k in FIELDS}
def make_state(arrays):
    s=ns.Mamba3State()
    for k,x in arrays.items(): setattr(s,k,nt(x))
    return s
def layer(seed=11,rank=1):
    c=ns.Mamba3Config();c.expand=1;c.head_dim=4;c.state_dim=128
    c.mimo=True;c.mimo_rank=rank;c.n_groups=1;c.seed=seed
    z=ns.Mamba3Layer(8,c);z.to(ns.Device.CPU)
    return z
def run(z,x,initial=None,valid=None):
    return z.forward_owned(nt(x),initial or ns.Mamba3State(),valid or [])
def objective(z,x,initial,dy,seeds):
    t=run(z,x,initial);out=t.output().numpy().copy()
    s=state_arrays(t.snapshot_final_state());t.cancel()
    return float(np.sum(out.astype(np.float64)*dy)+sum(np.sum(s[k].astype(np.float64)*seeds[k]) for k in FIELDS))

def smoke(outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(8021); details=[]
    for rank in (1,4):
        z=layer(rank=rank);x=rng.normal(0,.12,(2,5,8)).astype('f4')
        template=run(z,x[:,:1]);shapes=state_arrays(template.snapshot_final_state());template.cancel()
        initial={k:rng.normal(0,.01,v.shape).astype('f4') for k,v in shapes.items()}
        # Final phase is wrapped modulo 2*pi. A linear final-phase seed is
        # discontinuous at wrap; derivative fixtures stay inside one branch.
        initial['phase']+=2.
        seeds={k:rng.normal(0,.01,v.shape).astype('f4') for k,v in shapes.items()}
        dy=rng.normal(0,.1,x.shape).astype('f4');st=make_state(initial)
        t=run(z,x,st);full=t.output().numpy().copy();final=state_arrays(t.snapshot_final_state())
        g=z.backward_owned(t,nt(dy),make_state(seeds))
        assert t.audit_status()==[0,0]
        # Chronological chunk composition and cross-chunk boundary VJP.
        left=run(z,x[:,:2],st);right=run(z,x[:,2:],left.snapshot_final_state())
        chunk=np.concatenate([left.output().numpy(),right.output().numpy()],1)
        np.testing.assert_allclose(chunk,full,rtol=2e-5,atol=2e-6)
        rf=state_arrays(right.snapshot_final_state())
        for k in FIELDS: np.testing.assert_allclose(rf[k],final[k],rtol=2e-5,atol=2e-6)
        gr=z.backward_owned(right,nt(dy[:,2:]),make_state(seeds))
        gl=z.backward_owned(left,nt(dy[:,:2]),gr.initial_state)
        np.testing.assert_allclose(np.concatenate([gl.input.numpy(),gr.input.numpy()],1),g.input.numpy(),rtol=3e-4,atol=2e-6)
        for a,b,c in zip(g.parameters,gl.parameters,gr.parameters): np.testing.assert_allclose(b.numpy()+c.numpy(),a.numpy(),rtol=3e-4,atol=3e-6)
        for k in FIELDS: np.testing.assert_allclose(getattr(gl.initial_state,k).numpy(),getattr(g.initial_state,k).numpy(),rtol=3e-4,atol=3e-6)
        # Native FP32 directional derivatives, fixed tolerances, no torch port.
        errors={};eps=.002
        for name,value in [('input',x)]+list(initial.items()):
            d=rng.normal(size=value.shape).astype('f4');d/=np.linalg.norm(d)
            if name=='input':
                plus=objective(z,x+eps*d,st,dy,seeds);minus=objective(z,x-eps*d,st,dy,seeds);grad=g.input.numpy()
            else:
                plus=objective(z,x,make_state({**initial,name:value+eps*d}),dy,seeds)
                minus=objective(z,x,make_state({**initial,name:value-eps*d}),dy,seeds);grad=getattr(g.initial_state,name).numpy()
            numeric=(plus-minus)/(2*eps);analytic=float(np.sum(grad*d))
            print(json.dumps(dict(rank=rank,field=name,analytic=analytic,numeric=numeric)),flush=True)
            np.testing.assert_allclose(analytic,numeric,rtol=.02,atol=3e-4)
            errors[name]=dict(analytic=analytic,numeric=numeric,absolute_error=abs(analytic-numeric))
        # Binding exposes immutable weight snapshots; parameter VJP parity is
        # covered independently by the pinned upstream reference (eight cases).
        original=[p.data.numpy() for p in z.parameters()]
        # Masked NaN tails must be ignored and boundary of length zero retained.
        masked=x.copy();masked[0]=np.nan;masked[1,3:]=np.nan
        mt=run(z,masked,st,[0,3]);ms=state_arrays(mt.snapshot_final_state());mo=mt.output().numpy()
        assert mt.audit_status()==[0,0];assert np.isfinite(mo).all()
        for k in FIELDS: np.testing.assert_array_equal(ms[k][0],initial[k][0])
        mt.cancel()
        # One tape is single-use and output snapshots cannot mutate ownership.
        rejected=False
        try: z.backward_owned(t,nt(dy))
        except RuntimeError: rejected=True
        assert rejected
        details.append(dict(rank=rank,chunk_forward_max_abs=float(np.max(abs(chunk-full))),directional_vjp=errors,
                            masked_nan_pass=True,single_use_negative_pass=True,
                            configuration=z.configuration_identity(),parameter_count=sum(a.size for a in original),
                            persistent_state_bytes=sum(v.nbytes for v in initial.values())))
    result=dict(label='actual_native_Mamba3_CPU',passed=True,details=details,threads=2)
    (outdir/'native-equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

def features(src,dst,seed):
    data=np.load(src);z=layer(seed);result={};start=time.perf_counter()
    for name in data.files:
        x=data[name].astype('f4');t=run(z,x)
        assert t.audit_status()==[0]*len(x)
        result[name]=t.output().numpy().copy();t.cancel()
    np.savez(dst,**result)
    (dst.with_suffix('.json')).write_text(json.dumps(dict(label='actual_native_frozen_backbone_features',seed=seed,
        elapsed_seconds=time.perf_counter()-start,configuration=z.configuration_identity(),
        parameter_count=sum(p.data.numpy().size for p in z.parameters()),
        input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(dst.read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','features']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--seed',type=int,default=11);a=p.parse_args()
    smoke(a.out) if a.mode=='smoke' else features(a.input,a.out,a.seed)
