"""Run: python test_scan_recompute.py [--native-module-dir DIR] [--gpu]

Default never executes GPU. --gpu is exclusively for root's serialized hardware
gate, using the SAME independent oracle and artifacts, after integration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from mamba3_scan_oracle import (Geometry, affine_prefix, quadratic_states,
    operand_forward, operand_vjp, block_forward, zero_state, PIN)

RESULTS = []
NS = None
DEVICE = None
FIXTURES = None


def close(a,b,rtol=2e-11,atol=2e-12):
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)


def operand_fixture(T,R=4,P=3,N=128):
    rng = np.random.default_rng(1500+T+R)
    alpha = rng.uniform(.3,1,T)
    if T>3:
        alpha[1],alpha[3] = 0.,1.  # underflow and identity; never invert alpha
    return dict(alpha=alpha,beta=rng.uniform(0,.1,T),gamma=rng.uniform(0,.1,T),
        q=rng.normal(0,.1,(T,R,N)),k=rng.normal(0,.1,(T,R,N)),
        v=rng.normal(0,.1,(T,R,P)),h0=rng.normal(0,.1,(P,N)),
        k0=rng.normal(0,.1,(R,N)),v0=rng.normal(0,.1,(R,P)))


class AffineRecomputeTests(unittest.TestCase):
    def test_composition_order_noncommuting(self):
        alpha=np.array([.5,.25]);u=np.array([2.,4.]).reshape(2,1,1)
        close(affine_prefix(alpha,u,np.array([[4.]]),2).ravel(),[4.,5.])
        self.assertNotEqual(.5*4+2,.25*2+4)

    def test_n128_siso_mimo_scan_quadratic_serial(self):
        cases=0
        for R in (1,4,8):
          for T in (0,1,7,31,32,33,65,129):
            data=operand_fixture(T,R)
            ys,hs=operand_forward(**data,method='serial')
            yq,hq=operand_forward(**data,method='quadratic')
            close(ys,yq);close(hs,hq)
            for chunk in (1,7,16,32,64):
                ya,ha=operand_forward(**data,chunk=chunk)
                close(ya,yq);close(ha,hq);cases+=1
        RESULTS.append(dict(test='scan_serial_quadratic',cases=cases,N=128))

    def test_recompute_vjp_all_boundaries(self):
        cases=0
        for R in (1,4,8):
          for T in (0,1,31,32,33,65,129):
            d=operand_fixture(T,R)
            rng=np.random.default_rng(47)
            kwargs=dict(dy=rng.normal(0,.1,(T,R,3)),seed_h=rng.normal(0,.1,(3,128)),
                        seed_k=rng.normal(0,.1,(R,128)),seed_v=rng.normal(0,.1,(R,3)))
            full,full_size=operand_vjp(**d,**kwargs)
            for chunk in (1,7,16,32):
                replay,size=operand_vjp(**d,**kwargs,chunk=chunk)
                for key in full: close(replay[key],full[key])
                if T==129 and chunk==16:
                    self.assertLess(size,full_size)
                cases+=1
        RESULTS.append(dict(test='recompute_all_operand_adjoints',cases=cases,
                            storage_scope='SSM boundaries+local states only, not whole tape'))

    def test_operand_vjp_independent_finite_differences(self):
        d=operand_fixture(7)
        rng=np.random.default_rng(91)
        kwargs=dict(dy=rng.normal(0,.1,(7,4,3)),seed_h=rng.normal(0,.1,(3,128)),
                    seed_k=rng.normal(0,.1,(4,128)),seed_v=rng.normal(0,.1,(4,3)))
        def objective(p):
            y,h=operand_forward(**p,method='quadratic')
            return (np.sum(y*kwargs['dy'])+np.sum(h[-1]*kwargs['seed_h'])+
                    np.sum(p['k'][-1]*kwargs['seed_k'])+np.sum(p['v'][-1]*kwargs['seed_v']))
        grads,_=operand_vjp(**d,**kwargs,chunk=3)
        for key,value in d.items():
            direction=rng.normal(size=value.shape)
            direction/=np.linalg.norm(direction)
            step=2e-6
            plus={**d,key:value+step*direction};minus={**d,key:value-step*direction}
            numeric=(objective(plus)-objective(minus))/(2*step)
            close(np.sum(grads[key]*direction),numeric,rtol=2e-6,atol=2e-9)
        RESULTS.append(dict(test='quadratic_finite_difference_operand_vjp',directions=len(d)))

    def test_chunk_carry_kv_and_backward_seed(self):
        d=operand_fixture(33)
        y,h=operand_forward(**d,method='quadratic')
        cut=16
        left={key:(value[:cut] if key in ('alpha','beta','gamma','q','k','v') else value) for key,value in d.items()}
        right={key:(value[cut:] if key in ('alpha','beta','gamma','q','k','v') else value) for key,value in d.items()}
        yl,hl=operand_forward(**left)
        right.update(h0=hl[-1],k0=d['k'][cut-1],v0=d['v'][cut-1])
        yr,hr=operand_forward(**right)
        close(np.concatenate((yl,yr)),y);close(hr[-1],h[-1])
        rng=np.random.default_rng(2);dy=rng.normal(0,.1,y.shape)
        seeds=dict(seed_h=np.ones_like(d['h0'])*.03,seed_k=np.ones_like(d['k0'])*.02,seed_v=np.ones_like(d['v0'])*.04)
        whole,_=operand_vjp(**d,dy=dy,**seeds,chunk=7)
        gr,_=operand_vjp(**right,dy=dy[cut:],**seeds,chunk=7)
        gl,_=operand_vjp(**left,dy=dy[:cut],seed_h=gr['h0'],seed_k=gr['k0'],seed_v=gr['v0'],chunk=7)
        for key in ('alpha','beta','gamma','q','k','v'): close(np.concatenate((gl[key],gr[key])),whole[key])
        for key in ('h0','k0','v0'): close(gl[key],whole[key])
        # Mutations targeted at common chunk bugs MUST disagree with the oracle.
        wrong={**right,'k0':np.zeros_like(d['k0']),'v0':np.zeros_like(d['v0'])}
        yw,_=operand_forward(**wrong)
        self.assertGreater(np.max(np.abs(yw-yr)),1e-7)


def native_state(values):
    st=NS.Mamba3State()
    for key,value in values.items():
        setattr(st,key,NS.Tensor.from_numpy(value.astype(np.float32)).to(DEVICE))
    return st


def arrays_state(st):
    return {name:getattr(st,name).numpy().astype(np.float64) for name in ('phase','ssm','k','v')}


@unittest.skipIf(NS is None,'Native module selected in main before discovery')
class NativeLayerTests(unittest.TestCase):
    # skipIf binding evaluated before main; main sets __unittest_skip__ explicitly.
    def test_native_streaming_chunk_gradients_shared_groups(self):
        cfg=NS.Mamba3Config();cfg.expand=1;cfg.head_dim=2;cfg.n_groups=2;cfg.mimo=True
        cfg.mimo_rank=4;cfg.outproj_norm=True;cfg.rope_fraction=1.;cfg.seed=212
        layer=NS.Mamba3Layer(8,cfg);layer.to(DEVICE)
        g=Geometry(heads=4,groups=2,head_dim=2,rank=4,mimo=True,norm=True,rope=1.)
        rng=np.random.default_rng(701)
        x=rng.normal(0,.3,(2,65,8)).astype(np.float32);x[1,17:]=np.nan
        dy=rng.normal(0,.03,x.shape).astype(np.float32);dy[1,17:]=np.nan
        initial=zero_state(g,2)
        for name,val in initial.items(): initial[name]=rng.normal(0,.03,val.shape).astype(np.float32)
        seed={name:rng.normal(0,.02,val.shape).astype(np.float32) for name,val in initial.items()}
        tensor=lambda v:NS.Tensor.from_numpy(np.ascontiguousarray(v)).to(DEVICE)
        tape=layer.forward_owned(tensor(x),native_state(initial),[65,17])
        full_output=tape.output().numpy();full_final=arrays_state(tape.snapshot_final_state())
        whole=layer.backward_owned(tape,tensor(dy),native_state(seed))
        for cut in (16,32,33):
            left=layer.forward_owned(tensor(x[:,:cut]),native_state(initial),[cut,min(17,cut)])
            right=layer.forward_owned(tensor(x[:,cut:]),left.snapshot_final_state(),[65-cut,max(17-cut,0)])
            np.testing.assert_allclose(np.concatenate((left.output().numpy(),right.output().numpy()),axis=1),full_output,rtol=8e-4,atol=3e-5)
            for name,val in arrays_state(right.snapshot_final_state()).items(): np.testing.assert_allclose(val,full_final[name],rtol=8e-4,atol=3e-5)
            gr=layer.backward_owned(right,tensor(dy[:,cut:]),native_state(seed))
            gl=layer.backward_owned(left,tensor(dy[:,:cut]),gr.initial_state)
            np.testing.assert_allclose(np.concatenate((gl.input.numpy(),gr.input.numpy()),axis=1),whole.input.numpy(),rtol=8e-4,atol=3e-5)
            for name,val in arrays_state(gl.initial_state).items(): np.testing.assert_allclose(val,arrays_state(whole.initial_state)[name],rtol=8e-4,atol=3e-5)
            for gw,ga,gb in zip(whole.parameters,gl.parameters,gr.parameters): np.testing.assert_allclose(ga.numpy()+gb.numpy(),gw.numpy(),rtol=8e-4,atol=3e-5)
            self.assertEqual(left.audit_status(),[0,0]);self.assertEqual(right.audit_status(),[0,0])
        RESULTS.append(dict(test='native_streaming_shared_groups',boundaries=[16,32,33],lengths=[65,17]))

    def test_native_failure_cancel_foreign_tape(self):
        cfg=NS.Mamba3Config();cfg.expand=1;cfg.head_dim=4
        layer=NS.Mamba3Layer(8,cfg);layer.to(DEVICE)
        other=NS.Mamba3Layer(8,cfg);other.to(DEVICE)
        x=np.ones((1,3,8),np.float32)*.1
        tensor=lambda v:NS.Tensor.from_numpy(v).to(DEVICE)
        tape=layer.forward_owned(tensor(x))
        with self.assertRaises(Exception): other.backward_owned(tape,tensor(x))
        self.assertFalse(tape.consumed())
        tape.cancel()
        with self.assertRaises(Exception): layer.backward_owned(tape,tensor(x))
        bad=x.copy();bad[0,0,0]=np.inf
        tape=layer.forward_owned(tensor(bad))
        self.assertNotEqual(tape.audit_status(),[0]);close(tape.output().numpy(),np.zeros_like(x))
        grad=layer.backward_owned(tape,tensor(x))
        close(grad.input.numpy(),np.zeros_like(x))
        for value in grad.parameters: close(value.numpy(),np.zeros_like(value.numpy()))
        with self.assertRaises(Exception): layer.publish(grad)

    def test_full_layer_n128_outputs_states_directional_vjp(self):
        cases=0;max_error=0.;derivatives=0
        variants=[(False,1),(True,1),(True,4),(True,8)]
        for mimo,R in variants:
          for norm in (False,True):
           for rope in (.5,1.):
            g=Geometry(rank=R,mimo=mimo,norm=norm,rope=rope)
            cfg=NS.Mamba3Config();cfg.expand=1;cfg.head_dim=4;cfg.state_dim=128
            cfg.n_groups=1;cfg.mimo=mimo;cfg.mimo_rank=R;cfg.outproj_norm=norm;cfg.rope_fraction=rope;cfg.seed=117
            layer=NS.Mamba3Layer(8,cfg);layer.to(DEVICE)
            ps=layer.parameters();weights=[p.data.numpy().astype(np.float64) for p in ps]
            rng=np.random.default_rng(134+cases)
            x=rng.normal(0,.3,(2,33,8)).astype(np.float32).astype(np.float64)
            valid=[33,17];x[1,17:]=np.nan
            initial=zero_state(g,2)
            for name in initial:
                initial[name]=rng.normal(0,.025,initial[name].shape).astype(np.float32).astype(np.float64)
            # Wrap just beyond 2pi and negative phase; keep objective smooth.
            initial['phase'][:]=rng.uniform(-7.,7.,initial['phase'].shape).astype(np.float32)
            tape=layer.forward_owned(NS.Tensor.from_numpy(x.astype(np.float32)).to(DEVICE),native_state(initial),valid)
            want,fs=block_forward(x,weights,initial,valid,g,method='quadratic')
            scan,scanstate=block_forward(x,weights,initial,valid,g,chunk=16)
            close(scan,want);[close(scanstate[n],fs[n]) for n in fs]
            got=tape.output().numpy()
            np.testing.assert_allclose(got,want,rtol=8e-4,atol=3e-4)
            max_error=max(max_error,float(np.max(np.abs(got-want))))
            gotstate=arrays_state(tape.snapshot_final_state())
            for name in fs: np.testing.assert_allclose(gotstate[name],fs[name],rtol=8e-4,atol=3e-4,err_msg=name)
            dy=rng.normal(0,.03,want.shape).astype(np.float32).astype(np.float64);dy[1,17:]=np.nan
            seed={name:rng.normal(0,.02,val.shape).astype(np.float32).astype(np.float64) for name,val in fs.items()}
            grad=layer.backward_owned(tape,NS.Tensor.from_numpy(dy.astype(np.float32)).to(DEVICE),native_state(seed))
            self.assertEqual(tape.audit_status(),[0,0]);self.assertTrue(tape.consumed())
            with self.assertRaises(Exception): layer.backward_owned(tape,NS.Tensor.from_numpy(np.nan_to_num(dy).astype(np.float32)).to(DEVICE))
            actual=[grad.input.numpy().astype(np.float64)]+[p.numpy().astype(np.float64) for p in grad.parameters]
            inputs=[x]+weights
            stategrad=arrays_state(grad.initial_state)
            def objective(xx,ww,ss):
                out,final=block_forward(xx,ww,ss,valid,g,method='quadratic')
                safe_dy=np.nan_to_num(dy)
                return np.sum(out*safe_dy)+sum(np.sum(final[n]*seed[n]) for n in final)
            # Independent directions for input, EVERY registered parameter and all 4 initial fields.
            for index,(value,analytical) in enumerate(zip(inputs,actual)):
                direction=rng.normal(size=value.shape);direction/=np.linalg.norm(direction)
                if index==0: direction[1,17:]=0.
                eps=2e-5
                if index==0:
                    plus=objective(x+eps*direction,weights,initial);minus=objective(x-eps*direction,weights,initial)
                else:
                    wp=list(weights);wm=list(weights)
                    wp[index-1]=value+eps*direction;wm[index-1]=value-eps*direction
                    plus=objective(x,wp,initial);minus=objective(x,wm,initial)
                numeric=(plus-minus)/(2*eps)
                close(np.sum(analytical*direction),numeric,rtol=8e-4,atol=2e-7);derivatives+=1
            for name,value in initial.items():
                direction=rng.normal(size=value.shape);direction/=np.linalg.norm(direction)
                eps=2e-5
                numeric=(objective(x,weights,{**initial,name:value+eps*direction})-
                         objective(x,weights,{**initial,name:value-eps*direction}))/(2*eps)
                close(np.sum(stategrad[name]*direction),numeric,rtol=8e-4,atol=2e-7);derivatives+=1
            if FIXTURES:
                payload=dict(x=x,dy=dy,valid=np.array(valid),output=want,
                             native_dx=actual[0],geometry=np.array([R,mimo,norm,rope]))
                payload.update({f'w{i}':v for i,v in enumerate(weights)})
                payload.update({f'dw{i}':v for i,v in enumerate(actual[1:])})
                for name in fs:
                    payload.update({f'initial_{name}':initial[name],f'final_{name}':fs[name],
                                    f'seed_{name}':seed[name],f'dinitial_{name}':stategrad[name]})
                np.savez(FIXTURES/f'case-{cases:02}.npz',**payload)
            cases+=1
        RESULTS.append(dict(test='native_full_layer',device=str(DEVICE),cases=cases,
                            directional_derivatives=derivatives,max_abs_output=max_error))

    def test_empty_prefix_identity_and_masked_nan(self):
        g=Geometry(mimo=True,rank=4,norm=True,rope=1.)
        cfg=NS.Mamba3Config();cfg.expand=1;cfg.head_dim=4;cfg.mimo=True;cfg.mimo_rank=4;cfg.outproj_norm=True;cfg.rope_fraction=1.
        layer=NS.Mamba3Layer(8,cfg);layer.to(DEVICE)
        rng=np.random.default_rng(41);initial=zero_state(g,2)
        for name,val in initial.items(): initial[name]=rng.normal(0,.1,val.shape).astype(np.float32)
        x=np.full((2,7,8),np.nan,np.float32)
        tape=layer.forward_owned(NS.Tensor.from_numpy(x).to(DEVICE),native_state(initial),[0,0])
        self.assertEqual(tape.audit_status(),[0,0]);close(tape.output().numpy(),np.zeros_like(x))
        for name,val in arrays_state(tape.snapshot_final_state()).items(): close(val,initial[name])
        seed={name:rng.normal(0,.02,val.shape).astype(np.float32) for name,val in initial.items()}
        grad=layer.backward_owned(tape,NS.Tensor.from_numpy(x).to(DEVICE),native_state(seed))
        self.assertEqual(tape.audit_status(),[0,0]);close(grad.input.numpy(),np.zeros_like(x))
        for name,val in arrays_state(grad.initial_state).items(): close(val,seed[name])
        for val in grad.parameters: close(val.numpy(),np.zeros_like(val.numpy()))


def main():
    global NS,DEVICE,FIXTURES
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-module-dir',type=Path)
    parser.add_argument('--gpu',action='store_true',help='root-only serialized hardware gate')
    parser.add_argument('--result',type=Path,default=Path(__file__).with_name('cpu-evidence.json'))
    parser.add_argument('--fixtures-dir',type=Path)
    args=parser.parse_args()
    if args.gpu and not args.native_module_dir: parser.error('--gpu requires native module')
    if args.native_module_dir:
        sys.path.insert(0,str(args.native_module_dir.resolve()))
        dll_handles=[]
        if hasattr(os,'add_dll_directory'):
            for path in (args.native_module_dir,args.native_module_dir/'nsos',Path('C:/TheRock/build/bin')):
                if path.exists(): dll_handles.append(os.add_dll_directory(str(path.resolve())))
        import nsos_ext
        NS=nsos_ext;DEVICE=NS.Device.GPU if args.gpu else NS.Device.CPU
        NativeLayerTests.__unittest_skip__=False
    if args.fixtures_dir:
        FIXTURES=args.fixtures_dir;FIXTURES.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    info=dict(pin=PIN,python=sys.version,numpy=np.__version__,device='GPU' if args.gpu else 'CPU',
              elapsed_seconds=time.perf_counter()-start,tests=result.testsRun,
              passed=result.wasSuccessful(),skipped=len(result.skipped),details=RESULTS,
              gpu_executed=args.gpu,lds_executed=False)
    if NS:
        path=Path(NS.__file__)
        info.update(native_module=str(path),native_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args.result.write_text(json.dumps(info,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': sys.exit(main())
