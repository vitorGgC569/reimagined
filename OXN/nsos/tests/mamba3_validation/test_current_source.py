"""Compare isolated CPU compilation of CURRENT production math with NumPy oracle.

No prebuilt extension provenance assumption. Includes shared groups, padded NaN,
nonzero boundary states/seeds, A floor, zero/identity decay and RoPE layouts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from mamba3_scan_oracle import Geometry,zero_state,block_forward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args()
    stage=args.work_dir.resolve();stage.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(906)
    details=[];started=time.perf_counter()
    for case,(mimo,R,groups,rope,norm) in enumerate([
        (False,1,1,.5,False),(False,1,2,1.,True),
        (True,1,2,1.,False),(True,4,2,.5,True),(True,8,1,1.,True)]):
        B,S,H,P,N=3,33,4,3,128
        g=Geometry(heads=H,groups=groups,head_dim=P,rank=R,mimo=mimo,norm=norm,rope=rope)
        I=H*P;W=2*I+2*R*groups*N+3*H+g.pairs
        valid=np.array([0,17,33])
        projection=rng.normal(0,.25,(B,S,W))
        off=2*I+2*R*groups*N
        projection[2,1,off:off+H]=90.   # alpha underflows for positive A
        projection[2,2,off:off+H]=-90.  # alpha rounds to 1
        projection[2,3,off+H:off+2*H]=-20000.  # clamped A floor branch
        for b,length in enumerate(valid): projection[b,length:]=np.nan
        fields=[rng.uniform(.8,1.2,N),rng.uniform(.8,1.2,N),np.full(H,-1.),
                rng.normal(0,.05,(H,R,N)),rng.normal(0,.05,(H,R,N)),rng.uniform(.1,.3,H)]
        if mimo: fields.extend([rng.normal(.2,.1,(H,R,P)),rng.normal(.7,.1,(H,R,P)),rng.normal(.3,.1,(H,R,P))])
        if norm: fields.append(rng.uniform(.8,1.2,I))
        core=np.concatenate([a.ravel() for a in fields]);weights=[None,None]+fields
        initial=zero_state(g,B)
        for name,value in initial.items(): initial[name]=rng.normal(0,.04,value.shape)
        initial['phase']=rng.uniform(-12.,12.,initial['phase'].shape)
        seed={name:rng.normal(0,.03,value.shape) for name,value in initial.items()}
        dy=rng.normal(0,.02,(B,S,I))
        for b,length in enumerate(valid): dy[b,length:]=np.nan
        case_dir=stage/'current-source-fixtures'/f'case-{case:02}'
        case_dir.mkdir(parents=True,exist_ok=True)
        def write(name,value): np.asarray(value,dtype='<f8').tofile(case_dir/(name+'.bin'))
        for name,value in dict(projection=projection,core=core,dy=dy,valid=valid).items(): write(name,value)
        for name in initial: write('initial_'+name,initial[name]);write('seed_'+name,seed[name])
        shape=case_dir/'shape.txt';shape.write_text(f'{B} {S} {I} {H} {groups} {P} {N} {R} {g.pairs} {int(mimo)} {int(norm)}\n')
        subprocess.run([str(args.probe.resolve()),str(case_dir)+'/',str(shape)],check=True,capture_output=True)
        expected,final=block_forward(projection,weights,initial,valid,g,method='quadratic',projected=True)
        def read(name,shape): return np.fromfile(case_dir/(name+'.bin'),dtype='<f8').reshape(shape)
        errors={}
        for precision,tol in [('fp64',2e-10),('fp32',3e-4)]:
            assert np.all(read(precision+'_status',(B,))==0)
            got=read(precision+'_output',expected.shape)
            np.testing.assert_allclose(got,expected,rtol=8e-4 if precision=='fp32' else tol,atol=tol)
            errors[precision]=float(np.max(np.abs(got-expected)))
            for name,value in final.items():
                np.testing.assert_allclose(read(precision+'_final_'+name,value.shape),value,
                    rtol=8e-4 if precision=='fp32' else tol,atol=tol)
        def objective(raw,packed,st):
            unpacked=[];index=0
            for fld in fields:
                unpacked.append(packed[index:index+fld.size].reshape(fld.shape));index+=fld.size
            out,fs=block_forward(raw,[None,None]+unpacked,st,valid,g,method='quadratic',projected=True)
            return np.sum(out*np.nan_to_num(dy))+sum(np.sum(fs[name]*seed[name]) for name in fs)
        derivative_errors={}
        for name,value in [('projection',projection),('core',core)]+list(initial.items()):
            direction=rng.normal(size=value.shape);direction/=np.linalg.norm(direction)
            if name=='projection':
                for b,length in enumerate(valid): direction[b,length:]=0.
            step=2e-6
            if name=='projection': plus=objective(value+step*direction,core,initial);minus=objective(value-step*direction,core,initial)
            elif name=='core': plus=objective(projection,value+step*direction,initial);minus=objective(projection,value-step*direction,initial)
            else: plus=objective(projection,core,{**initial,name:value+step*direction});minus=objective(projection,core,{**initial,name:value-step*direction})
            numeric=(plus-minus)/(2*step)
            output_name={'projection':'dx','core':'dp'}.get(name,'dinitial_'+name)
            for precision,tol in [('fp64',2e-8),('fp32',2e-5)]:
                analytical=np.sum(read(precision+'_'+output_name,value.shape)*direction)
                np.testing.assert_allclose(analytical,numeric,rtol=8e-4 if precision=='fp32' else 2e-6,atol=tol)
                derivative_errors[precision+'_'+name]=float(abs(analytical-numeric))
        details.append(dict(case=case,mimo=mimo,rank=R,groups=groups,rope=rope,norm=norm,
                            max_abs_output=errors,derivative_errors=derivative_errors))
    info=dict(passed=True,device='CPU',gpu_executed=False,source_compiled=True,cases=len(details),
              elapsed_seconds=time.perf_counter()-started,details=details,
              exe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest())
    (stage/'current-source-evidence.json').write_text(json.dumps(info,indent=2)+'\n')
    print(f'CURRENT SOURCE CPU INDEPENDENT ORACLE PASS cases={len(details)} FP64/FP32 N128 shared groups/boundaries',flush=True)


if __name__=='__main__': main()
