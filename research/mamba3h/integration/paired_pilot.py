"""Bounded actual-native frozen-backbone factorial; no synthetic Mamba port.

Optional branches are auxiliary modules, not a replacement of native SSM math.
Input residual is common to all arms. Paired initialization/minibatch schedules.
Native process never imports torch; training process never imports nsos_ext.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
AREA=ROOT/'research/mamba3h'
VENDOR=AREA/'integration/vendor'
sys.path.insert(0,str(VENDOR if (VENDOR/'snapshot.json').exists() else ROOT))
if str(ROOT) not in sys.path:sys.path.append(str(ROOT))
PROTOCOL_FILE=AREA/'manifests/pilot-v1.json'
PROTOCOL=json.loads(PROTOCOL_FILE.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')

def prepare(out):
    from research.mamba3h.benchmarks.generation import generate
    from research.mamba3h.benchmarks.schema import encode_numeric,model_view,fingerprint
    manifest=[]
    for seed in PROTOCOL['seeds']:
        sd=out/f'seed-{seed}';sd.mkdir(parents=True,exist_ok=True)
        inputs={};episodes={};splits={}
        for split,count in PROTOCOL['samples'].items():
            eps=generate('inst',seed,split,count,**PROTOCOL['difficulty']);episodes[split]=eps
            inputs[split]=np.asarray([encode_numeric(e,8) for e in eps],dtype='f4')
            splits[split]=[fingerprint(model_view(e)) for e in eps]
        assert all(not set(splits[a]) & set(splits[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
        from research.mamba3h.benchmarks.oracle import solve
        for split,eps in episodes.items():
            for ep in eps:assert solve(ep)==ep['targets'],'independent solver mismatch'
        dump(sd/'episodes.json',episodes);np.savez(sd/'input.npz',**inputs)
        cmd=[sys.executable,str(AREA/'integration/native_cpu.py'),'features','--input',str(sd/'input.npz'),'--out',str(sd/'native.npz'),'--seed',str(seed)]
        t=time.perf_counter();r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=90)
        (sd/'native.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
        manifest.append(dict(seed=seed,split_fingerprints=splits,episodes_sha256=sha(sd/'episodes.json'),
            input_sha256=sha(sd/'input.npz'),native_sha256=sha(sd/'native.npz'),elapsed=time.perf_counter()-t))
    dump(out/'dataset-manifest.json',manifest)
    print('PREPARED five paired actual native feature datasets',flush=True)

def model_classes():
    import torch
    from torch import nn
    from research.mamba3h.algebra.operators import StructuredTransition
    from research.mamba3h.memory.slots import CausalSlotMemory,SparseRetrievalAttention,Control
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    class Model(nn.Module):
        def __init__(self,arm,seed):
            super().__init__();self.arm=arm
            self.algebra=StructuredTransition(dim=8,rank=1,operations=10,commuting=arm=='MC',seed=seed,dtype=torch.float32) if arm in ('M1','M3','MC') else None
            # Explicit per-module seeds avoid arm-dependent readout initialization.
            torch.manual_seed(seed+1000)
            self.memory=(SparseRetrievalAttention if arm=='MA' else CausalSlotMemory)(8,4,1) if arm in ('M2','M3','MA') else None
            torch.manual_seed(seed+2000);self.head=nn.Linear(8,6)
        def forward(self,x,events):
            batch,length,_=x.shape;ast=None;mst=self.memory.initial_state(batch) if self.memory else None
            ys=[];norms=[];counts={};cache=0
            for t in range(length):
                z=x[:,t];y=z
                if self.algebra:
                    ids=torch.tensor([int(e[t]['operator'] or 0) for e in events],dtype=torch.long)
                    # This auxiliary recurrence has global per-example state;
                    # it is not an oracle S5 solver or native state update.
                    previous=ast if ast is not None else torch.zeros_like(z)
                    ay,candidate,_=self.algebra.step(z,previous,control={'op_id':ids})
                    op_mask=torch.tensor([e[t]['kind']=='OP' for e in events],dtype=torch.bool)[:,None]
                    ast=torch.where(op_mask,candidate,previous+z)
                    y=ast;norms.extend(ast.detach().norm(dim=-1).tolist())
                if self.memory:
                    current=[e[t] for e in events];entity=torch.tensor([e['entity'] for e in current],dtype=torch.long)
                    write=torch.tensor([e['kind'] in ('SET','OP','WRITE') for e in current],dtype=torch.bool)
                    read=torch.tensor([e['kind']=='QUERY' for e in current],dtype=torch.bool)
                    revoke=torch.tensor([e['entity'] if e['kind']=='REVOKE' else -1 for e in current],dtype=torch.long)
                    if self.arm=='MA':
                        write=torch.ones(batch,dtype=torch.bool);entity=torch.full((batch,),-1,dtype=torch.long);revoke=torch.full((batch,),-1,dtype=torch.long)
                    # v1 intentionally preserved parallel branches. v2 is a
                    # causal wiring correction: structured state BEFORE read.
                    pre_state=ast if self.arm=='M3' and PROTOCOL.get('memory_input_M3')=='algebra_pre_read' else z
                    my,mst,stats=self.memory.step(pre_state,mst,Control(write,read,entity,revoke))
                    y=y+(my-pre_state)
                    for name in ('reads','similarity_evaluations','similarity_madds','topk_calls','value_reads','writes','evictions','overwrites','revocations'):
                        counts[name]=counts.get(name,0)+stats[name]
                    cache=max(cache,stats['cache_bytes_estimate'])
                    norms.extend(mst.values.detach().flatten(1).norm(dim=-1).tolist())
                ys.append(self.head(y))
            telemetry=dict(counts=counts,cache_bytes_estimate=cache,aux_state_bytes=(mst.tensor_bytes() if mst else 0)+(batch*8*4 if ast is not None else 0),
                state_norm_max=max(norms,default=0),state_norm_p99=float(np.quantile(norms,.99)) if norms else 0)
            return torch.stack(ys,1),telemetry
    return Model,torch

def train(out,seed,arm):
    Model,torch=model_classes();sd=out/f'seed-{seed}';episodes=json.loads((sd/'episodes.json').read_text())
    inputs=np.load(sd/'input.npz');native=np.load(sd/'native.npz')
    x={k:torch.tensor(inputs[k]+native[k]) for k in inputs.files}
    targets={k:torch.tensor([[5 if v==125 else v for v in e['targets']] for e in eps]) for k,eps in episodes.items()}
    events={k:[e['events'] for e in eps] for k,eps in episodes.items()}
    model=Model(arm,seed);opt=torch.optim.Adam(model.parameters(),lr=.01)
    rng=np.random.default_rng(seed+3000);gradnorm=[];losses=[];start=time.perf_counter();ever_nonzero=set()
    for step in range(PROTOCOL['updates']):
        ix=rng.choice(len(x['train']),PROTOCOL['minibatch'],replace=False);index=torch.tensor(ix)
        opt.zero_grad(set_to_none=True);logits,_=model(x['train'][index],[events['train'][i] for i in ix])
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets['train'][index].flatten(),ignore_index=-100)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for name,p in model.named_parameters():
            if p.grad is not None and bool((p.grad!=0).any()):ever_nonzero.add(name)
        if not torch.isfinite(g):raise FloatingPointError('nonfinite gradient')
        opt.step();losses.append(float(loss.detach()));gradnorm.append(float(g))
    evaluations={};model.eval()
    with torch.no_grad():
        for split in ('validation','test'):
            t=time.perf_counter();correct=total=0;norms=[];counts={};maxcache=0;state_bytes=0
            for i in range(0,len(x[split]),8):
                logits,st=model(x[split][i:i+8],events[split][i:i+8]);y=targets[split][i:i+8];mask=y!=-100
                correct+=int(((logits.argmax(-1)==y)&mask).sum());total+=int(mask.sum())
                for name,count in st['counts'].items():counts[name]=counts.get(name,0)+count
                norms.append(st['state_norm_max']);maxcache=max(maxcache,st['cache_bytes_estimate']);state_bytes=max(state_bytes,st['aux_state_bytes'])
            evaluations[split]=dict(accuracy=correct/total,correct=correct,total=total,latency_seconds=time.perf_counter()-t,
                retrieval=counts,aux_state_bytes_per_eval_batch=state_bytes,cache_bytes_estimate=maxcache,state_norm_max=max(norms))
    result=dict(status='completed',seed=seed,arm=arm,label=PROTOCOL['label'],updates=len(losses),elapsed_seconds=time.perf_counter()-start,
        loss_first=losses[0],loss_last=losses[-1],grad_norm_max=max(gradnorm),grad_norm_p99=float(np.quantile(gradnorm,.99)),
        trainable_parameters=sum(p.numel() for p in model.parameters()),trainable_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        persistent_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),evaluations=evaluations,
        parameters_ever_nonzero_gradient=sum(p.numel() for name,p in model.named_parameters() if name in ever_nonzero),
        native_frozen_parameters=3340,native_persistent_state_bytes_per_example=5408,
        feature_cache_bytes={k:int(inputs[k].nbytes+native[k].nbytes) for k in inputs.files},
        routing='visible causal metadata, no learned gates',native_backbone_frozen=True,full_compute_matched=False,
        protocol=PROTOCOL['protocol'],protocol_sha256=sha(PROTOCOL_FILE),
        M3_memory_input=PROTOCOL.get('memory_input_M3','native_pre_read'),runner_sha256=sha(Path(__file__).resolve()))
    dump(sd/f'{arm}.json',result);print(json.dumps(result),flush=True)

def run_all(out):
    start=time.perf_counter();ledger=[]
    for seed in PROTOCOL['seeds']:
        for arm in PROTOCOL['arms']:
            job=dict(seed=seed,arm=arm)
            if arm in PROTOCOL.get('reuse_arms',[]):
                reuse=json.loads((out/'reuse-manifest.json').read_text())
                entry=next(e for e in reuse['cells'] if e['seed']==seed and e['arm']==arm)
                assert sha(out/f'seed-{seed}'/f'{arm}.json')==entry['sha256']
                job.update(status='completed',origin='reused unchanged pilot-v1 cell',source_sha256=entry['sha256'],wall_seconds=0.)
                ledger.append(job);dump(out/'run-ledger.json',ledger);continue
            if time.perf_counter()-start>=PROTOCOL['aggregate_learned_wall_seconds']:
                job.update(status='unattempted',reason='aggregate time budget exhausted');ledger.append(job);continue
            cmd=[sys.executable,str(Path(__file__).resolve()),'train','--out',str(out),'--seed',str(seed),'--arm',arm,'--protocol',str(PROTOCOL_FILE)]
            then=time.perf_counter()
            try:
                r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=min(90,900-(time.perf_counter()-start)))
                (out/f'seed-{seed}'/f'{arm}.log').write_text(r.stdout+r.stderr)
                job.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
            except subprocess.TimeoutExpired as e:
                job.update(status='censored',reason='wall time limit');(out/f'seed-{seed}'/f'{arm}.log').write_text(str(e))
            job['wall_seconds']=time.perf_counter()-then;ledger.append(job)
            dump(out/'run-ledger.json',ledger);print(json.dumps(job),flush=True)
    dump(out/'run-ledger.json',ledger)

def summarize(out):
    ledger=json.loads((out/'run-ledger.json').read_text());records=[]
    for job in ledger:
        path=out/f"seed-{job['seed']}"/f"{job['arm']}.json"
        if job['status']=='completed':records.append(json.loads(path.read_text()))
    by={(r['seed'],r['arm']):r for r in records};comparisons={}
    for arm in PROTOCOL['arms'][1:]:
        pairs=[(by[s,arm]['evaluations']['test']['accuracy'],by[s,'M0']['evaluations']['test']['accuracy']) for s in PROTOCOL['seeds'] if (s,arm) in by and (s,'M0') in by]
        d=np.array([a-b for a,b in pairs]);n=len(d)
        comparisons[arm]=dict(paired_count=n,deltas=d.tolist(),mean_delta=float(d.mean()) if n else None,
            interval_95_t_df4=[float(d.mean()-2.776445105*np.std(d,ddof=1)/np.sqrt(5)),float(d.mean()+2.776445105*np.std(d,ddof=1)/np.sqrt(5))] if n==5 else None,
            five_seed_claim_allowed=n==5,gap_recovery=None,gap_reason='oracle ceiling not measured')
    interactions=[by[s,'M3']['evaluations']['test']['accuracy']-by[s,'M1']['evaluations']['test']['accuracy']-by[s,'M2']['evaluations']['test']['accuracy']+by[s,'M0']['evaluations']['test']['accuracy'] for s in PROTOCOL['seeds'] if all((s,a) in by for a in ('M0','M1','M2','M3'))]
    result=dict(protocol=PROTOCOL,label=PROTOCOL['label'],ledger=ledger,records=records,paired_vs_M0=comparisons,
        optional_raw_accuracy_interaction=interactions,claim_scope='small frozen native backbone plus parallel auxiliary adapters; not native Mamba3H',
        full_compute_matched=False,oracle_ceiling_measured=False,gpu_executed=False,
        source_grid_reuse=PROTOCOL.get('reuse_arms',[]),parent_grid=PROTOCOL.get('parent_grid'))
    dump(out/'summary.json',result);print(json.dumps(comparisons,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','run','summarize']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--protocol',type=Path,default=PROTOCOL_FILE)
    a=p.parse_args();a.out=a.out.resolve();PROTOCOL_FILE=a.protocol.resolve();PROTOCOL=json.loads(PROTOCOL_FILE.read_text())
    {'prepare':lambda:prepare(a.out),'train':lambda:train(a.out,a.seed,a.arm),'run':lambda:run_all(a.out),'summarize':lambda:summarize(a.out)}[a.mode]()
