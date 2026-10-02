"""Post-run reporting only: no hyperparameter/model/test selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse,json,numpy as np
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h';OUT=AREA/'review/pilot'
def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();OUT=args.out.resolve()
    s=json.loads((OUT/'summary.json').read_text());diagnostics=[]
    for seed in s['protocol']['seeds']:
        sd=OUT/f'seed-{seed}';eps=json.loads((sd/'episodes.json').read_text())
        labels={split:np.array([v for e in episodes for v in e['targets'] if v!=-100]) for split,episodes in eps.items()}
        counts=np.bincount(labels['train'],minlength=126);majority=int(counts.argmax());f=np.load(sd/'native.npz')
        diagnostics.append(dict(seed=seed,train_majority_class=majority,
            train_selected_constant_test_accuracy=float(np.mean(labels['test']==majority)),
            uniform_6_class_reference=1/6,test_label_counts={str(k):int(v) for k,v in enumerate(np.bincount(labels['test'],minlength=126)) if v},
            native_feature_max_abs={split:float(abs(f[split]).max()) for split in f.files},
            native_feature_norm_p99={split:float(np.quantile(np.linalg.norm(f[split],axis=-1),.99)) for split in f.files}))
    rows=[]
    for arm in s['protocol']['arms']:
        r=[x for x in s['records'] if x['arm']==arm];ev=[x['evaluations']['test'] for x in r]
        rows.append(dict(arm=arm,accuracy_mean=float(np.mean([e['accuracy'] for e in ev])),
            accuracy_by_seed=[e['accuracy'] for e in ev],trainable_parameters=r[0]['trainable_parameters'],
            effective_gradient_parameter_count_by_seed=[x['parameters_ever_nonzero_gradient'] for x in r],
            native_frozen_parameters=3340,native_state_bytes_per_example=5408,
            aux_state_bytes_per_example=ev[0]['aux_state_bytes_per_eval_batch']//8,
            test_latency_seconds_mean=float(np.mean([e['latency_seconds'] for e in ev])),
            state_norm_max=max(e['state_norm_max'] for e in ev),preclip_gradient_norm_max=max(x['grad_norm_max'] for x in r),
            retrieval_by_seed=[e['retrieval'] for e in ev],peak_cache_bytes_estimate=max(e['cache_bytes_estimate'] for e in ev)))
    d=np.array(s['optional_raw_accuracy_interaction']);interaction=dict(mean=float(d.mean()),by_seed=d.tolist(),
        interval_95_t_df4=[float(d.mean()-2.776445105*d.std(ddof=1)/np.sqrt(5)),float(d.mean()+2.776445105*d.std(ddof=1)/np.sqrt(5))])
    result=dict(rows=rows,conditioning_and_constant_controls=diagnostics,interaction=interaction,
        learned_wall_seconds=sum(j['wall_seconds'] for j in s['ledger']),
        memory_caveat='Hard top-k1 softmax is constant and selection has no STE: Q/K scoring gradients are zero. Gates unused. This pilot does not establish learned addressing.',
        algebra_caveat='Auxiliary global state per example, not entity-scoped group state or native transition. NOOP has identity homogeneous operator and visible input injection.',
        matching='M1/MC equal raw parameter count; M2/MA same initialized modules and capacity. M3/M0 counts/active state/compute differ. No matched whole-grid compute claim.',
        conditioning_caveat='Native outputs are unnormalized, and large additive states/preclip gradients were observed. No post-test conditioning or update retuning was performed.',
        M3_memory_input=s['protocol'].get('memory_input_M3','native_pre_read'),
        corrected_grid_note='v2 preserves/reuses 25 unchanged v1 cells and executes five M3 causal-wiring corrections' if s['protocol'].get('reuse_arms') else 'v1 original full grid preserved')
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
