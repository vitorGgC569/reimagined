"""Audit historical 42/15 log; optionally execute real seeded native CPU trials.

--run-full uses unmodified source functions/main, except explicit global/native
seeds and identical per-model Flappy shuffle streams. This is paired_shuffle_v1,
NOT an assertion that undocumented historical RNG or 42/15 can be recovered.
--smoke runs repeated, bounded Flappy train/eval through the SAME native classes.
No source edits, no GPU, no mocks. Results have module/source/data/order hashes.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict,List,Tuple
import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def datahash(data):
    return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()


def definitions(source,ns):
    tree=ast.parse(source.read_text(encoding='utf-8'))
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    module=ast.Module(body=body,type_ignores=[])
    context=dict(math=math,os=os,random=random,sys=sys,time=time,Path=Path,
                 Dict=Dict,List=List,Tuple=Tuple,np=np,nsos_ext=ns)
    exec(compile(module,str(source),'exec',flags=__import__('__future__').annotations.compiler_flag),context)
    return context


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--historical-log',type=Path,required=True)
    p.add_argument('--native-module-dir',type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run-full',action='store_true');mode.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=117)
    p.add_argument('--result',type=Path,default=Path(__file__).with_name('empirical-evidence.json'))
    args=p.parse_args()
    source=args.repo/'tests/test_mamba2_vs_mamba3_real.py'
    text=args.historical_log.read_text(encoding='utf-8')
    info=dict(protocol='paired_shuffle_v1',source=str(source),source_sha256=sha(source),
              historical_log=str(args.historical_log),historical_log_sha256=sha(args.historical_log),
              seed=args.seed,python=sys.version,numpy=np.__version__,gpu_executed=False,
              historical_rng_recoverable=False,historical_reproduced=False,
              audit=dict(flappy_max={'Mamba-2':15,'Mamba-3':42},
                         flappy_mean={'Mamba-2':8.6,'Mamba-3':19.5},
                         cpu_train_seconds={'Mamba-2':113.1,'Mamba-3':169.5},
                         max_is_not_mean=True,flappy_examples_are_frames_not_trajectories=True,
                         reversal_target='last input token only; not full sequence reversal',
                         missing_historical_provenance=['global shuffle seed','native initialization seed',
                             'module/source SHA','per-trial scores','epoch sample order','build flags']),
              lr={'Mamba-2':.015,'Mamba-3':.002},native_executed=False)
    required=('Avg Score: 8.6 pipes | Max Score: 15','Avg Score: 19.5 pipes | Max Score: 42')
    if not all(x in text for x in required): raise RuntimeError('historical log metrics differ; re-audit required')
    if not (args.run_full or args.smoke):
        args.result.write_text(json.dumps(info,indent=2)+'\n');print('Historical provenance audit PASS; no native execution requested',flush=True);return
    if args.native_module_dir is None: p.error('native module required for execution')
    # Deny optimized/GPU provider policies in this CPU runner.
    if os.environ.get('NSOS_MAMBA3_GPU_PROVIDER','dense_reference')!='dense_reference':
        raise RuntimeError('CPU replay requires dense_reference provider')
    sys.path.insert(0,str(args.native_module_dir.resolve()))
    import nsos_ext as ns
    ctx=definitions(source,ns)
    info.update(native_executed=True,native_module=str(Path(ns.__file__)),native_sha256=sha(Path(ns.__file__)),
                omp_threads=os.environ.get('OMP_NUM_THREADS'),runs=[],shuffle_sha256=[],initializations=[])
    original_shuffle=random.shuffle
    def logged_shuffle(data):
        original_shuffle(data);info['shuffle_sha256'].append(datahash(data))
    random.shuffle=logged_shuffle
    for factory_name in ('build_mamba2_model','build_mamba3_model'):
        original=ctx[factory_name]
        def factory(*values,_original=original,_name=factory_name,**kwargs):
            ns.set_seed(args.seed)
            model,count=_original(*values,**kwargs)
            params=model.parameters()
            digest=hashlib.sha256()
            for param in params: digest.update(param.data.numpy().tobytes())
            info['initializations'].append(dict(factory=_name,parameters=count,sha256=digest.hexdigest()))
            return model,count
        ctx[factory_name]=factory
    started=time.perf_counter()
    if args.run_full:
        # Record genuine source trial functions; keep all original native train calls.
        for fn in ('run_flappy_trial','run_associative_recall_trial','run_sequence_reversal_trial'):
            original=ctx[fn]
            def trial(*values,_original=original,_name=fn,**kwargs):
                random.seed(args.seed);np.random.seed(args.seed)
                result=_original(*values,**kwargs)
                info['runs'].append(dict(task=_name,model=values[0],metrics=result))
                args.result.write_text(json.dumps(info,indent=2)+'\n')
                return result
            ctx[fn]=trial
        random.seed(args.seed);np.random.seed(args.seed);ns.set_seed(args.seed)
        ctx['main']()
        if info['shuffle_sha256'][:6]!=info['shuffle_sha256'][6:12]:
            raise AssertionError('paired shuffle differs between architectures')
        info['profile']='full original tasks; paired Flappy shuffle'
    else:
        env=ctx['FlappyEnvironment'](seed=777);dataset=[]
        while len(dataset)<64:
            obs=env.reset();done=False
            while not done and len(dataset)<64:
                act=ctx['flappy_expert_policy'](obs)
                dataset.append((ctx['FlappyTokenizer'].encode_state(obs),[61 if act else 60]))
                obs,_,done,_=env.step(act)
        info['dataset_sha256']=datahash(dataset)
        for repeat in range(2):
          for name,factory_name,lr in [('Mamba-2','build_mamba2_model',.015),('Mamba-3','build_mamba3_model',.002)]:
            random.seed(args.seed);np.random.seed(args.seed)
            model,count=ctx[factory_name](2,128,64)
            trainer=ns.Trainer(model,lr);trainer.weight_decay=.0001;trainer.max_grad_norm=1.
            working=list(dataset);losses=[]
            for epoch in range(2):
                random.shuffle(working)
                losses=[float(trainer.train_supervised(prompt,target)) for prompt,target in working]
            agent=ctx['NSOSAgent'](model);scores=[];frames=[]
            for trial_index in range(10):
                env=ctx['FlappyEnvironment'](seed=555+trial_index)
                obs=env.reset();done=False
                while not done and env.frames<2000:
                    obs,_,done,score=env.step(agent.predict_action(obs))
                scores.append(score);frames.append(env.frames)
            trained=hashlib.sha256()
            for param in model.parameters(): trained.update(param.data.numpy().tobytes())
            row=dict(model=name,repeat=repeat,parameters=count,final_loss=float(np.mean(losses)),
                     scores=scores,frames=frames,trained_sha256=trained.hexdigest())
            info['runs'].append(row);print(json.dumps(row),flush=True)
        for i in (0,1):
            a,b=info['runs'][i],info['runs'][i+2]
            for key in ('final_loss','scores','frames','trained_sha256'):
                if a[key]!=b[key]: raise AssertionError(f'CPU determinism mismatch {a["model"]}: {key}')
        if len(set(info['shuffle_sha256'][0::2]))!=1 or len(set(info['shuffle_sha256'][1::2]))!=1:
            raise AssertionError('smoke paired sample order mismatch')
        info.update(profile='smoke64 frames, 2 epochs, 10 independent trial seeds, repeated twice',
                    deterministic_repeated_native_cpu=True)
    random.shuffle=original_shuffle
    info['elapsed_seconds']=time.perf_counter()-started
    args.result.write_text(json.dumps(info,indent=2)+'\n')
    print('REAL NATIVE CPU REPLAY COMPLETE; historical 42/15 is not an acceptance threshold',flush=True)


if __name__=='__main__': main()
