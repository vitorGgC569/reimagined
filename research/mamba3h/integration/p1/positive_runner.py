"""Fresh held-out native-frozen positive control, separate from frozen P0.

No test tuning, no native port/GPU. Explicit symbolic address and anonymous k2
are distinguished; sufficient attention slots eliminate forced-eviction confound.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h'
PROTOCOL_FILE=AREA/'manifests/p1a-positive-r2.json';P=json.loads(PROTOCOL_FILE.read_text())
VENDOR=AREA/'integration/vendor'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')
def numeric(ep):
    rows=[]
    for ev in ep['events']:
        assert ev['kind'] in ('WRITE','NOOP','QUERY')
        row=np.zeros(8,dtype='f4');row[ev['entity']]=1
        if ev['kind']=='WRITE':row[4+ev['value']]=1
        rows.append(row)
    return np.stack(rows)

def prepare(out,corpus):
    assert not out.exists(),'Use a new output directory; preserve all previous attempts'
    manifest=[];all_inputs=set()
    for seed in P['seeds']:
        sd=out/f'seed-{seed}';sd.mkdir(parents=True,exist_ok=True);arrays={};episodes={};entries=[]
        for split,count in P['samples'].items():
            src=corpus/f'seed-{seed}'/f'{split}.jsonl'
            eps=[json.loads(line) for line in src.read_text().splitlines()];assert len(eps)==count
            for ep in eps:
                assert len(ep['events'])==6 and ep['events'][-1]['kind']=='QUERY'
                writes=[e for e in ep['events'][:5] if e['kind']=='WRITE']
                assert len(writes)==3 and sum(e['kind']=='NOOP' for e in ep['events'][:5])==2
                assert len({e['entity'] for e in writes})==3
                q=ep['events'][-1]['entity'];values={e['entity']:e['value'] for e in writes}
                assert ep['targets']==[-100]*5+[values[q]],'Root2 independent replay check'
                x=numeric(ep);h=hashlib.sha256(x.tobytes()).hexdigest()
                assert h not in all_inputs,'Cross-split/seed duplicate input';all_inputs.add(h)
            arrays[split]=np.stack([numeric(ep) for ep in eps]);episodes[split]=eps
            entries.append(dict(split=split,source=str(src.relative_to(ROOT)),source_sha256=sha(src),examples=count))
        put(sd/'episodes.json',episodes);np.savez(sd/'input.npz',**arrays)
        cmd=[sys.executable,'-B',str(AREA/'integration/p1/native_provider.py'),'features','--input',str(sd/'input.npz'),'--out',str(sd/'native.npz'),'--seed',str(seed)]
        r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=90);(sd/'native.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
        n=np.load(sd/'native.npz');raw_rms=np.sqrt(np.mean(arrays['train']**2,axis=(0,1),dtype=np.float64))
        native_rms=np.sqrt(np.mean(n['train']**2,axis=(0,1),dtype=np.float64))
        # Zero channels retain unit scale, with explicit mask; no test-derived eps.
        raw_scale=np.where(raw_rms>0,raw_rms,1);native_scale=np.where(native_rms>0,native_rms,1)
        put(sd/'conditioning.json',dict(source_split='train',labels_used=False,raw_rms=raw_rms.tolist(),native_rms=native_rms.tolist(),
            raw_scale=raw_scale.tolist(),native_scale=native_scale.tolist(),raw_train_sha256=hashlib.sha256(arrays['train'].tobytes()).hexdigest(),
            native_train_sha256=hashlib.sha256(n['train'].tobytes()).hexdigest()))
        manifest.append(dict(seed=seed,splits=entries,input_sha256=sha(sd/'input.npz'),native_sha256=sha(sd/'native.npz'),
                             conditioning_sha256=sha(sd/'conditioning.json')))
    put(out/'dataset-manifest.json',dict(protocol_sha256=sha(PROTOCOL_FILE),provider_sha256=P['provider_sha256'],unique_inputs=len(all_inputs),seeds=manifest))
    print('Prepared800 fresh positive examples, independent replay and unique input checks PASS',flush=True)

def classes():
    sys.path.insert(0,str(VENDOR))
    import torch
    from torch import nn
    from research.mamba3h.memory.slots import CausalSlotMemory,SparseRetrievalAttention,Control,COUNTERS
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    class Model(nn.Module):
        def __init__(self,arm,seed):
            super().__init__();self.arm=arm;self.memory=None;self.ff1=None
            torch.manual_seed(seed+1000)
            if arm.startswith(('M2','MA')):self.memory=(SparseRetrievalAttention if arm.startswith('MA') else CausalSlotMemory)(8,6,2)
            if arm=='MC_capacity':
                self.ff1=nn.Linear(8,14);self.ff2=nn.Linear(14,8);self.gate=nn.Linear(8,4);self.scale=nn.Parameter(torch.ones(()))
            torch.manual_seed(seed+2000);self.head=nn.Linear(8,4)
            if self.memory:assert self.memory.initial_state(1).tensor_bytes()==630
            if arm!='M0':assert sum(p.numel() for p in self.parameters())==319
        def forward(self,x,events):
            batch,length,_=x.shape;state=self.memory.initial_state(batch) if self.memory else None
            padding=torch.zeros(batch,630,dtype=torch.uint8) if self.memory is None else None
            outputs=[];totals={k:0 for k in COUNTERS};peakcache=0;norms=[];candidates=[]
            for t in range(length):
                z=x[:,t];y=z
                if self.ff1 is not None:
                    y=z+self.scale*torch.tanh(self.ff2(torch.tanh(self.ff1(z))))*torch.sigmoid(self.gate(z)).repeat(1,2)
                if self.memory:
                    ev=[e[t] for e in events];read=torch.tensor([e['kind']=='QUERY' for e in ev])
                    write=torch.tensor([e['kind']=='WRITE' or self.arm.startswith('MA') for e in ev])
                    entity=torch.tensor([e['entity'] if self.arm=='M2_symbolic_address' else -1 for e in ev],dtype=torch.long)
                    y,state,stats=self.memory.step(z,state,Control(write,read,entity))
                    for k in COUNTERS:totals[k]+=stats[k]
                    peakcache=max(peakcache,stats['cache_bytes_estimate']);norms.extend(state.values.detach().flatten(1).norm(dim=-1).tolist())
                    if t==5:candidates=stats['selected_indices']
            # Loss/readout is at the one final query only; native/all memory
            # operations still process all six observed current tokens.
            return self.head(y),dict(counters=totals,reserved_aux_bytes=batch*630,active_aux_bytes=state.tensor_bytes() if state else 0,
                dummy_padding_bytes=padding.numel() if padding is not None else 0,cache_bytes_estimate=peakcache,
                slot_norm_max=max(norms,default=0),slot_norm_p99=float(np.quantile(norms,.99)) if norms else 0,
                selected_indices=candidates)
    return Model,torch

def ready(out):
    assert sha(ROOT/P['provider_path'])==P['provider_sha256'],'Private frozen provider drift'
    receipt=json.loads((out/'dataset-manifest.json').read_text())
    assert receipt['protocol_sha256']==sha(PROTOCOL_FILE) and receipt['unique_inputs']==800
    assert receipt['provider_sha256']==P['provider_sha256']
    assert [r['seed'] for r in receipt['seeds']]==P['seeds']
    for row in receipt['seeds']:
        sd=out/f"seed-{row['seed']}"
        for filename,key in [('input.npz','input_sha256'),('native.npz','native_sha256'),('conditioning.json','conditioning_sha256')]:
            assert sha(sd/filename)==row[key],'Prepared feature/conditioning drift'

def train(out,seed,arm):
    ready(out)
    assert seed in P['seeds'] and arm in P['arms']
    assert not (out/f'seed-{seed}'/f'{arm}-initial.npz').exists(),'Never overwrite a trained attempt'
    Model,torch=classes();sd=out/f'seed-{seed}';ep=json.loads((sd/'episodes.json').read_text());raw=np.load(sd/'input.npz');native=np.load(sd/'native.npz')
    cond=json.loads((sd/'conditioning.json').read_text());rs=np.array(cond['raw_scale']);ns=np.array(cond['native_scale'])
    x={k:torch.tensor((raw[k]/rs+native[k]/ns).astype('f4')) for k in raw.files}
    targets={k:torch.tensor([e['targets'][-1] for e in eps]) for k,eps in ep.items()};events={k:[e['events'] for e in eps] for k,eps in ep.items()}
    m=Model(arm,seed);initial={name:v.detach().numpy().copy() for name,v in m.state_dict().items()};np.savez(sd/f'{arm}-initial.npz',**initial)
    opt=torch.optim.Adam(m.parameters(),lr=P['learning_rate']);rng=np.random.default_rng(seed+3000);losses=[];gradnorm=[];active=set();qkg=[];start=time.perf_counter()
    for step in range(P['updates']):
        ids=rng.choice(len(x['train']),P['minibatch'],replace=False);ix=torch.tensor(ids)
        opt.zero_grad(set_to_none=True);logits,_=m(x['train'][ix],[events['train'][i] for i in ids]);loss=torch.nn.functional.cross_entropy(logits,targets['train'][ix])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        for name,p in m.named_parameters():
            if p.grad is not None and bool((p.grad!=0).any()):active.add(name)
        if m.memory:
            qkg.append({name:float(getattr(m.memory,name).weight.grad.norm()) if getattr(m.memory,name).weight.grad is not None else 0 for name in ('query','key')})
        g=torch.nn.utils.clip_grad_norm_(m.parameters(),P['gradient_clip']);assert torch.isfinite(g),'nonfinite gradient'
        opt.step();losses.append(float(loss.detach()));gradnorm.append(float(g))
    m.eval();evaluations={}
    with torch.no_grad():
        for split in ('validation','test'):
            before=time.perf_counter();correct=0;total=0;ct={};cache=0;norm=0;reserved=active_bytes=padding=0
            for i in range(0,len(x[split]),8):
                logits,st=m(x[split][i:i+8],events[split][i:i+8]);y=targets[split][i:i+8]
                correct+=int((logits.argmax(-1)==y).sum());total+=len(y)
                for k,v in st['counters'].items():ct[k]=ct.get(k,0)+v
                cache=max(cache,st['cache_bytes_estimate']);norm=max(norm,st['slot_norm_max']);reserved=max(reserved,st['reserved_aux_bytes']);active_bytes=max(active_bytes,st['active_aux_bytes']);padding=max(padding,st['dummy_padding_bytes'])
            assert ct['evictions']==0,'C6 positive comparator must not force eviction'
            evaluations[split]=dict(accuracy=correct/total,correct=correct,total=total,latency_seconds=time.perf_counter()-before,
                actual_counters=ct,reserved_aux_bytes_per_example=reserved//8,active_aux_bytes_per_example=active_bytes//8,
                dummy_padding_bytes_per_example=padding//8,cache_bytes_estimate=cache,slot_norm_max=norm)
    np.savez(sd/f'{arm}-final.npz',**{name:v.detach().numpy().copy() for name,v in m.state_dict().items()})
    result=dict(status='completed',seed=seed,arm=arm,label=P['label'],updates=len(losses),elapsed_seconds=time.perf_counter()-start,
        losses=losses,grad_norm_max=max(gradnorm),grad_norm_p99=float(np.quantile(gradnorm,.99)),qk_gradient_norms=qkg,
        gradient_active_parameter_tensor_sizes=sum(p.numel() for name,p in m.named_parameters() if name in active),
        parameters=sum(p.numel() for p in m.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),
        evaluations=evaluations,protocol_sha256=sha(PROTOCOL_FILE),runner_sha256=sha(Path(__file__).resolve()),
        initial_weights_sha256=sha(sd/f'{arm}-initial.npz'),final_weights_sha256=sha(sd/f'{arm}-final.npz'),native_frozen_parameters=3340,
        native_state_bytes_per_example=5408,routing='symbolic ADDRESS' if arm=='M2_symbolic_address' else 'anonymous k2' if m.memory else 'no auxiliary retrieval',
        common_conditioning_sha256=sha(sd/'conditioning.json'),compute_fully_matched=False)
    put(sd/f'{arm}.json',result);print(json.dumps(dict(seed=seed,arm=arm,test_accuracy=evaluations['test']['accuracy'],qk_gradient_active=any(v['query']>0 and v['key']>0 for v in qkg))),flush=True)

def run(out):
    ready(out)
    assert not (out/'run-ledger.json').exists(),'Never overwrite an executed ledger'
    start=time.perf_counter();ledger=[]
    for seed in P['seeds']:
        for arm in P['arms']:
            remaining=min(P['new_learned_wall_budget_seconds']-(time.perf_counter()-start),P['cumulative_root2_learned_wall_budget_seconds']-P['prior_root2_learned_wall_seconds']-(time.perf_counter()-start))
            job=dict(seed=seed,arm=arm)
            if remaining<=0:job.update(status='unattempted',reason='aggregate declared time budget exhausted',wall_seconds=0);ledger.append(job);continue
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'train','--out',str(out),'--seed',str(seed),'--arm',arm]
            before=time.perf_counter()
            try:
                r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=min(P['per_job_seconds'],remaining));(out/f'seed-{seed}'/f'{arm}.log').write_text(r.stdout+r.stderr)
                job.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
            except subprocess.TimeoutExpired as e:job.update(status='censored',reason='declared wall timeout');(out/f'seed-{seed}'/f'{arm}.log').write_text(str(e))
            job['wall_seconds']=time.perf_counter()-before;ledger.append(job);put(out/'run-ledger.json',ledger);print(json.dumps(job),flush=True)
    put(out/'run-ledger.json',ledger)

def summarize(out):
    ledger=json.loads((out/'run-ledger.json').read_text());records=[]
    for j in ledger:
        if j['status']=='completed':records.append(json.loads((out/f"seed-{j['seed']}"/f"{j['arm']}.json").read_text()))
    table={(r['seed'],r['arm']):r for r in records};cmp={}
    for candidate,baseline in [(a,'M0') for a in P['arms'][1:]]+[('M2_anonymous_k2','MA_anonymous_k2'),('M2_symbolic_address','M2_anonymous_k2'),('M2_anonymous_k2','MC_capacity')]:
        delta=[table[s,candidate]['evaluations']['test']['accuracy']-table[s,baseline]['evaluations']['test']['accuracy'] for s in P['seeds'] if (s,candidate) in table and (s,baseline) in table]
        n=len(delta);d=np.array(delta);mean=float(d.mean()) if n else None
        cmp[candidate+' - '+baseline]=dict(complete_pairs=n,deltas=delta,mean_delta=mean,
            interval_95_t_df4=[float(mean-2.776445105*d.std(ddof=1)/np.sqrt(5)),float(mean+2.776445105*d.std(ddof=1)/np.sqrt(5))] if n==5 else None)
    wall=sum(j['wall_seconds'] for j in ledger)
    # Exact initialized Q/K/V/output/gates/head arrays must match memory arms.
    init_checks=[]
    for seed in P['seeds']:
        paths=[out/f'seed-{seed}'/f'{a}-initial.npz' for a in P['arms'] if a.startswith(('M2','MA'))]
        if all(p.exists() for p in paths):
            arrays=[np.load(p) for p in paths]
            for name in arrays[0].files:
                for arr in arrays[1:]:np.testing.assert_array_equal(arrays[0][name],arr[name])
            init_checks.append(dict(seed=seed,bitwise_initialized_memory_and_head=True))
    result=dict(protocol=P,ledger=ledger,records=records,paired_comparisons=cmp,initialization_checks=init_checks,
        learned_wall_seconds=wall,cumulative_root2_learned_wall_seconds=wall+P['prior_root2_learned_wall_seconds'],
        original_setup_failures=P['original_setup_failures'],original_setup_unattempted=P['original_setup_unattempted'],
        setup_failure_conservative_charge_seconds=P['setup_failure_conservative_charge_seconds'],
        full_compute_matching=False,claim_scope=P['claim_scope'],gap_recovery=None,
        gap_reason='symbolic assistance is a separate routing condition, not an admissible learned-addressing ceiling')
    put(out/'summary.json',result);print(json.dumps(dict(comparisons=cmp,wall=wall),indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','train','run','summarize']);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--corpus',type=Path);a.add_argument('--seed',type=int);a.add_argument('--arm');args=a.parse_args();out=args.out.resolve()
    {'prepare':lambda:prepare(out,args.corpus.resolve()),'train':lambda:train(out,args.seed,args.arm),'run':lambda:run(out),'summarize':lambda:summarize(out)}[args.mode]()
