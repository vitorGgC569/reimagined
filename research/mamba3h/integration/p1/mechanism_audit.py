"""Declared posthoc zero-update checkpoint interventions; no tuning or new trials."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from positive_runner import classes,ready,sha,AREA,P
DATA=AREA/'review/p1/positive-r2';OUT=AREA/'review/p1/mechanism'
assert not OUT.exists();OUT.mkdir(parents=True)
PROTOCOL=AREA/'manifests/p1a-mechanism.json';p=json.loads(PROTOCOL.read_text())
summary=json.loads((DATA/'summary.json').read_text());ready(DATA)
Model,torch=classes();start=time.perf_counter();results=[];ledger=[]
for seed in p['seeds']:
    sd=DATA/f'seed-{seed}';episodes=json.loads((sd/'episodes.json').read_text())['test']
    raw=np.load(sd/'input.npz')['test'];native=np.load(sd/'native.npz')['test'];cond=json.loads((sd/'conditioning.json').read_text())
    x=torch.tensor((raw/np.array(cond['raw_scale'])+native/np.array(cond['native_scale'])).astype('f4'))
    targets=torch.tensor([e['targets'][-1] for e in episodes]);events=[e['events'] for e in episodes]
    for arm in p['arms']:
        initial=np.load(sd/f'{arm}-initial.npz');final=np.load(sd/f'{arm}-final.npz')
        for variant in p['variants']:
            job=dict(seed=seed,arm=arm,variant=variant)
            if time.perf_counter()-start>p['wall_budget_seconds']:
                ledger.append(dict(**job,status='unattempted',reason='declared budget exhausted'));continue
            m=Model(arm,seed);state={name:torch.tensor(final[name]) for name in final.files}
            if variant=='restore_initial_QK_keep_final_values_output_head':
                for name in ('memory.query.weight','memory.key.weight'):state[name]=torch.tensor(initial[name])
            if variant=='zero_retrieval_output_keep_final_head':state['memory.output.weight']=torch.zeros_like(state['memory.output.weight'])
            m.load_state_dict(state,strict=True);m.eval();correct=hits=0;count=0
            with torch.no_grad():
                for i in range(0,64,8):
                    logits,stats=m(x[i:i+8],events[i:i+8]);correct+=int((logits.argmax(-1)==targets[i:i+8]).sum())
                    for b,evs in enumerate(events[i:i+8]):
                        q=evs[-1]['entity'];write_t=next(t for t,e in enumerate(evs[:5]) if e['kind']=='WRITE' and e['entity']==q)
                        slot=write_t if arm.startswith('MA') else sum(e['kind']=='WRITE' for e in evs[:write_t])
                        hits+=int(slot in stats['selected_indices'][b]);count+=1
            accuracy=correct/count
            if variant=='full_checkpoint_replay':
                original=next(r for r in summary['records'] if r['seed']==seed and r['arm']==arm)
                assert accuracy==original['evaluations']['test']['accuracy'],'Checkpoint replay changed original outcome'
            results.append(dict(**job,accuracy=accuracy,correct=correct,total=count,selected_original_write_fraction=hits/count,
                final_checkpoint_sha256=sha(sd/f'{arm}-final.npz'),initial_checkpoint_sha256=sha(sd/f'{arm}-initial.npz')))
            ledger.append(dict(**job,status='completed',optimizer_updates=0))
table={(r['seed'],r['arm'],r['variant']):r for r in results};comparisons=[]
for arm in p['arms']:
    for variant in p['variants'][1:]:
        deltas=[table[s,arm,p['variants'][0]]['accuracy']-table[s,arm,variant]['accuracy'] for s in p['seeds'] if (s,arm,variant) in table and (s,arm,p['variants'][0]) in table]
        d=np.array(deltas);mean=float(d.mean()) if len(d) else None
        interval=[float(mean-2.776445105*d.std(ddof=1)/np.sqrt(5)),float(mean+2.776445105*d.std(ddof=1)/np.sqrt(5))] if len(d)==5 else None
        comparisons.append(dict(arm=arm,full_minus=variant,complete_pairs=len(d),deltas=deltas,mean_delta=mean,interval_95_t_df4_unadjusted=interval))
receipt=dict(protocol=p,protocol_sha256=sha(PROTOCOL),source_sha256=sha(Path(__file__)),results=results,ledger=ledger,
    paired_comparisons=comparisons,elapsed_seconds=time.perf_counter()-start,optimizer_updates=0,threads=2)
(OUT/'results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(paired_comparisons=comparisons,elapsed_seconds=receipt['elapsed_seconds']),indent=2))
