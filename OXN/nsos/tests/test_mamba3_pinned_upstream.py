"""Execute the pinned upstream *CPU chunk reference*, not TileLang/CuTe kernels.

Requires torch and einops. Supply --module-dir and --artifact-dir. Extract only
the original reference functions by AST; do not import GPU backend test fixtures.
This distinguishes executable primary-source reference parity from kernel
parity on upstream's NVIDIA platform (not claimed by this test).
"""
from pathlib import Path
import argparse,ast,hashlib,sys,typing,math,os
import numpy as np
import subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module-dir',type=Path,required=True)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--native-phase',choices=('export','check'),help=argparse.SUPPRESS)
    parser.add_argument('--fixtures-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    sys.path.insert(0,str(args.artifact_dir/'mamba3-test-deps'))
    sys.path.insert(0,str(args.module_dir))
    dlls=[]
    if hasattr(os,'add_dll_directory'):
        for p in (args.module_dir,args.module_dir/'nsos',Path('C:/TheRock/build/bin')):
            if p.exists(): dlls.append(os.add_dll_directory(str(p.resolve())))
    variants=[(rank,rope,norm) for rank in (1,4) for rope in (.5,1.) for norm in (False,True)]
    fixtures=args.fixtures_dir or args.artifact_dir/('upstream-fixtures-hip' if args.gpu else 'upstream-fixtures-cpu')
    fixtures.mkdir(parents=True,exist_ok=True)
    if args.native_phase:
        # Native process never imports torch: avoid loading conflicting OpenMP runtimes.
        import nsos_ext as ns
        dev=ns.Device.GPU if args.gpu else ns.Device.CPU
        max_error=0.
        for case,(rank,rope,norm) in enumerate(variants):
            c=ns.Mamba3Config();c.expand=1;c.head_dim=4;c.n_groups=1;c.state_dim=128;c.mimo=True;c.mimo_rank=rank;c.rope_fraction=rope;c.outproj_norm=norm
            layer=ns.Mamba3Layer(8,c);layer.to(dev);ps=layer.parameters()
            path=fixtures/f'case-{case}.npz'
            if args.native_phase=='export':
                np.savez(path,**{f'w{i}':p.data.numpy() for i,p in enumerate(ps)})
                continue
            data=np.load(path)
            for i,p in enumerate(ps):
                np.testing.assert_array_equal(p.data.numpy(),data[f'w{i}'])
            tape=layer.forward_owned(ns.Tensor.from_numpy(data['input']).to(dev))
            got=tape.output().numpy();want=data['output']
            np.testing.assert_allclose(got,want,rtol=8e-4,atol=3e-4)
            max_error=max(max_error,float(np.max(np.abs(got-want))))
            gradients=layer.backward_owned(tape,ns.Tensor.from_numpy(data['dy']).to(dev))
            np.testing.assert_allclose(gradients.input.numpy(),data['dx'],rtol=8e-4,atol=3e-4)
            for i,(p,g) in enumerate(zip(ps,gradients.parameters)):
                np.testing.assert_allclose(g.numpy(),data[f'dw{i}'],rtol=8e-4,atol=3e-4,err_msg=p.name)
            assert tape.audit_status()==[0]
        if args.native_phase=='check':
            print(f'PINNED UPSTREAM CPU REFERENCE PASS cases={len(variants)} native_device={"GPU" if args.gpu else "CPU"} max_abs_output={max_error:.9g}',flush=True)
        return
    command=[sys.executable,str(Path(__file__).resolve()),'--module-dir',str(args.module_dir),'--artifact-dir',str(args.artifact_dir),'--fixtures-dir',str(fixtures)]
    if args.gpu: command.append('--gpu')
    subprocess.run(command+['--native-phase','export'],check=True)
    import torch,einops
    torch.set_num_threads(1)
    # Torch CPU primary-source oracle stays in this process; native checks run in a child.
    source=args.artifact_dir/'upstream-mamba3/test_mamba3_mimo.py'
    raw=source.read_bytes()
    expected_sha256='2259f1f32b4c58ef3ed2976076bd4ab6a50382a9902a16fbfbac404a1fde3618'
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise RuntimeError('Pinned upstream source SHA256 mismatch')
    original=ast.parse(raw.decode())
    names={'mamba3_MIMO_chunk_ref','_pad_zeros'}
    extracted=ast.Module(body=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    assert len(extracted.body)==2
    context={'torch':torch,'Tensor':torch.Tensor,'Optional':typing.Optional,'F':torch.nn.functional,'math':math,'rearrange':einops.rearrange,'repeat':einops.repeat}
    exec(compile(extracted,str(source),'exec'),context)
    upstream=context['mamba3_MIMO_chunk_ref']
    cases=0
    for rank in (1,4):
      for rope in (.5,1.):
       for norm in (False,True):
        data=dict(np.load(fixtures/f'case-{cases}.npz'))
        weights=[torch.tensor(data[f'w{i}'],dtype=torch.float64,requires_grad=True) for i in range(len(data))]
        eps=1e-5;floor=1e-4
        H=2;P=4;N=128;G=1;R=rank;I=8;A=int(N*rope/2);B=1;S=4
        array=(np.sin(np.arange(B*S*8)*.27)*.13).astype(np.float32).reshape(B,S,8)
        x=torch.tensor(array,dtype=torch.float64,requires_grad=True)
        proj=x @ weights[0].T
        z,v,k,q,dt,a,trap,angle=torch.split(proj,[I,I,R*G*N,R*G*N,H,H,H,A],dim=-1)
        k=k.reshape(B,S,R,G,N);q=q.reshape(B,S,R,G,N)
        k=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+eps)*weights[2]
        q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+eps)*weights[3]
        dt=torch.nn.functional.softplus(dt+weights[4]);a=-torch.clamp(torch.where(a>=0,1+a,1/(1-a)),min=floor)
        adt=(a*dt).transpose(1,2)
        dacs=adt.cumsum(-1);dacs_rev=adt.flip(-1).cumsum(-1).flip(-1)-adt
        phase=(torch.tanh(angle).unsqueeze(2).expand(B,S,H,A)*dt.unsqueeze(-1)*math.pi).cumsum(1)
        y,_,_=upstream(q,k,v.reshape(B,S,H,P),weights[6],weights[5],weights[8],weights[10],z.reshape(B,S,H,P),weights[9],phase,dacs,dacs_rev,dt.transpose(1,2),trap.transpose(1,2),weights[7],chunk_size=S,rotary_dim_divisor=int(2/rope),dtype=torch.float64,fused_norm=norm,outproj_norm_weight=weights[11] if norm else None,outproj_norm_eps=eps)
        # Upstream returns FP32 readout even with dtype=float64; keep original reference unchanged.
        expected=y.to(torch.float64).reshape(B,S,I)@weights[1].T
        dy=(np.cos(np.arange(B*S*8)*.19)*.01).astype(np.float32).reshape(B,S,8)
        (expected*torch.tensor(dy)).sum().backward()
        data.update(input=array,dy=dy,output=expected.detach().numpy(),dx=x.grad.numpy())
        for i,w in enumerate(weights):
            assert w.grad is not None,f'parameter {i}'
            data[f'dw{i}']=w.grad.numpy()
        np.savez(fixtures/f'case-{cases}.npz',**data)
        cases+=1
    subprocess.run(command+['--native-phase','check'],check=True)
    print(f'pin=e9594ce1c732d97440f0332fdc43170a2294dbfa upstream_source_sha256={hashlib.sha256(raw).hexdigest()} torch={torch.__version__} einops={einops.__version__}')

if __name__=='__main__': main()
