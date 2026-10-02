"""Predeclared five-seed isolated projection/readout feasibility probe, CPU only."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from torch import nn
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from .slots import CausalSlotMemory, SparseRetrievalAttention, Control, route_oracle, COUNTERS

HERE=Path(__file__).resolve().parent


def examples(seed,count,protocol):
    g=torch.Generator().manual_seed(seed)
    xs=torch.zeros(count,6,8)
    targets=torch.zeros(count,dtype=torch.long)
    for b in range(count):
        ids=torch.randperm(4,generator=g)[:3]
        values=torch.randint(4,(3,),generator=g)
        for t in range(3):
            xs[b,t,ids[t]]=1
            xs[b,t,4+values[t]]=1
        for t in (3,4):
            xs[b,t,int(torch.randint(4,(1,),generator=g))]=1
        which=int(torch.randint(3,(1,),generator=g))
        xs[b,5,ids[which]]=1
        targets[b]=values[which]
    return xs,targets


class Probe(nn.Module):
    def __init__(self,arm,p):
        super().__init__()
        cls=CausalSlotMemory if arm=='memory_WR' else SparseRetrievalAttention
        self.memory=cls(p['dim'],p['capacity'],p['top_k'])
        self.readout=nn.Linear(p['dim'],p['value_classes'])
        self.arm=arm

    def forward(self,x):
        batch=x.shape[0]
        state=self.memory.initial_state(batch)
        stats=None
        max_state=0.
        max_cache=0
        for t in range(x.shape[1]):
            base=Control(torch.full((batch,),t<3,dtype=torch.bool),
                         torch.full((batch,),t==5,dtype=torch.bool))
            control=route_oracle('store_all',base) if self.arm=='sparse_store_all' else base
            y,state,stats=self.memory.step(x[:,t],state,control)
            max_state=max(max_state,float(state.values.detach().norm(dim=-1).max()))
            max_cache=max(max_cache,stats['cache_bytes_estimate'])
        # Target tensor is deliberately absent from this interface.
        return self.readout(y),state,dict(stats,max_slot_norm=max_state,max_cache_bytes_estimate=max_cache)


def fp(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    protocol_path=HERE/'trial_protocol.json'
    p=json.loads(protocol_path.read_text())
    receipt={'protocol':p,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
             'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('__init__.py','slots.py','learned_trial.py')},
             'preimport_thread_environment':{name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
             'torch_version':torch.__version__,'threads':torch.get_num_threads(),
             'interop_threads':torch.get_num_interop_threads(),'jobs':[],
             'claim_scope':'isolated_learned_projection_and_readout_only'}
    output=HERE/'learned_results.json'
    start=time.perf_counter()
    for seed in p['seeds']:
        data={split:examples(seed+offset,count,p) for split,offset,count in
              (('train',10000,p['train_examples']),('validation',20000,p['validation_examples']),('test',30000,p['test_examples']))}
        for arm in p['arms']:
            job={'seed':seed,'arm':arm,'status':'unattempted','updates':0,
                 'splits':{split:{'generator_seed':seed+off,'input_sha256':fp(data[split][0]),'target_sha256':fp(data[split][1])}
                           for split,off in (('train',10000),('validation',20000),('test',30000))}}
            receipt['jobs'].append(job)
            if time.perf_counter()-start>=p['aggregate_seconds']:
                job['reason']='aggregate budget exhausted'
                continue
            job_start=time.perf_counter()
            try:
                torch.manual_seed(seed)
                model=Probe(arm,p)
                optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
                job.update(status='running',parameter_count=sum(q.numel() for q in model.parameters()),
                           parameter_bytes=sum(q.numel()*q.element_size() for q in model.parameters()),
                           inactive_gate_parameters=sum(q.numel() for q in model.memory.gates.parameters()),
                           initial_model_sha256=hashlib.sha256(b''.join(q.detach().numpy().tobytes() for q in model.parameters())).hexdigest())
                max_grad=0.
                max_norm=0.
                for update in range(p['updates']):
                    if time.perf_counter()-job_start>p['per_job_seconds'] or time.perf_counter()-start>p['aggregate_seconds']:
                        job.update(status='censored',reason='time budget exhausted')
                        break
                    optimizer.zero_grad(set_to_none=True)
                    logits,state,stats=model(data['train'][0])
                    loss=nn.functional.cross_entropy(logits,data['train'][1])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    grads=[q.grad for q in model.parameters() if q.grad is not None]
                    if not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError('nonfinite gradient')
                    max_grad=max(max_grad,float(torch.sqrt(sum((g*g).sum() for g in grads))))
                    max_norm=max(max_norm,stats['max_slot_norm'])
                    optimizer.step()
                    job['updates']=update+1
                    job['final_training_loss']=float(loss.detach())
                else:
                    job['status']='complete'
                job.update(max_gradient_norm=max_grad,max_slot_norm_training=max_norm)
                if job['status']=='complete':
                    with torch.no_grad():
                        for split,(xs,targets) in data.items():
                            eval_start=time.perf_counter()
                            logits,state,stats=model(xs)
                            if not torch.isfinite(logits).all():
                                raise FloatingPointError('nonfinite evaluation')
                            job[split]={'accuracy':float((logits.argmax(-1)==targets).float().mean()),
                                        'loss':float(nn.functional.cross_entropy(logits,targets)),
                                        'total_operations':{name:int(state.counters[:,i].sum()) for i,name in enumerate(COUNTERS)},
                                        'state_bytes':state.tensor_bytes(),
                                        'cache_bytes_estimate':stats['max_cache_bytes_estimate'],
                                        'occupancy':stats['occupancy'],
                                        'latency_seconds':time.perf_counter()-eval_start,
                                        'max_slot_norm':stats['max_slot_norm']}
            except Exception as exc:
                job.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            job['elapsed_seconds']=time.perf_counter()-job_start
            receipt['elapsed_seconds']=time.perf_counter()-start
            output.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({k:job[k] for k in ('seed','arm','status','updates','elapsed_seconds')},sort_keys=True),flush=True)
    paired=[]
    for seed in p['seeds']:
        rows={j['arm']:j for j in receipt['jobs'] if j['seed']==seed}
        if all(rows[a]['status']=='complete' for a in p['arms']):
            paired.append({'seed':seed,'memory_minus_sparse':rows['memory_WR']['test']['accuracy']-rows['sparse_store_all']['test']['accuracy']})
    receipt['paired_deltas']=paired
    if len(paired)==5:
        ds=[r['memory_minus_sparse'] for r in paired]
        mean=statistics.mean(ds)
        half=2.7764451051977987*statistics.stdev(ds)/(5**.5)
        receipt['paired_t95_df4']={'mean':mean,'lower':mean-half,'upper':mean+half}
    else:
        receipt['paired_t95_df4']=None
    receipt['gap_recovery']=None
    receipt['gap_recovery_reason']='no frozen native baseline/ceiling comparison in isolated trial'
    receipt['elapsed_seconds']=time.perf_counter()-start
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':receipt['elapsed_seconds'],'paired_t95_df4':receipt['paired_t95_df4']}),flush=True)


if __name__=='__main__':
    main()
