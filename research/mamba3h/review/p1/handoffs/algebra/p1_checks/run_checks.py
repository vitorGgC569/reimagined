"""Bounded no-learning checks and separate evidence, P0 bytes preserved."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import time
import unittest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
import torch
from exact_entity import ExactEntityControl,run_events
import test_exact_entity
from research.mamba3h.benchmarks.generation import generate
from research.mamba3h.benchmarks.oracle import solve
from research.mamba3h.benchmarks.schema import IGNORE,UNDEFINED,canonical_bytes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def p0_hashes():
    p0=HERE.parent
    frozen=json.loads((p0/'manifest.json').read_text())
    names=list(frozen['files'])+['manifest.json']
    return {name:sha(p0/name) for name in names}


def main():
    start=time.perf_counter();before=p0_hashes()
    protocol=json.loads((HERE/'protocol.json').read_text())
    results=dict(scope=protocol['scope'],device='CPU',torch_threads=torch.get_num_threads(),
                 learned=False,optimizer_updates=0,protocol_sha256=sha(HERE/'protocol.json'),
                 preserved_p0_before=before,cells=[],unattempted=protocol['unattempted'])
    with (HERE/'tests.log').open('w',encoding='utf-8') as log:
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_exact_entity)
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    results['tests']=dict(run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors))
    for difficulty in protocol['grid']:
        for seed in protocol['paired_seeds']:
            for split in protocol['splits']:
                base=dict(seed=seed,split=split,difficulty=difficulty)
                if not result.wasSuccessful() or time.perf_counter()-start>protocol['wall_seconds']:
                    results['cells'].append(dict(base,status='unattempted_correctness_or_budget'))
                    continue
                episodes=generate('inst',seed,split,protocol['episodes_per_seed_split'],group='s5',regime='within_group',**difficulty)
                labels=[solve(ep) for ep in episodes] # offline, never supplied to model
                cell=dict(base,status='executed',episodes=len(episodes),
                    corpus_sha256=hashlib.sha256(canonical_bytes([ep['events'] for ep in episodes])).hexdigest(),
                    solver_labels_sha256=hashlib.sha256(canonical_bytes(labels)).hexdigest(),arms={})
                for arm,per_entity in (('global',False),('per_entity',True)):
                    model=ExactEntityControl(difficulty['entities'],per_entity)
                    correct=0;total=0;whole=0;failures=[];tail=[];counts={};undefined_queries=0
                    arm_start=time.perf_counter()
                    for index,(ep,oracle) in enumerate(zip(episodes,labels)):
                        predictions,states,stats=run_events(ep['events'],model)
                        expected=[None if v==IGNORE else 5 if v==UNDEFINED else v for v in oracle]
                        whole+=int(predictions==expected)
                        for t,(pred,label) in enumerate(zip(predictions,expected)):
                            if label is None:continue
                            total+=1;correct+=int(pred==label);undefined_queries+=int(label==5)
                            if pred!=label:failures.append(dict(episode=index,event_index=t,predicted=pred,expected=label))
                        tail.extend(float(st.norm()) for st in states)
                        for stat in stats:
                            for k,v in stat['counts'].items():counts[k]=counts.get(k,0)+v
                    _,_,budget=model.step([ep['events'][0]])
                    cell['arms'][arm]=dict(correct=correct,queries=total,accuracy=correct/total if total else None,
                        whole_episode_correct=whole,undefined_queries=undefined_queries,failures=failures,
                        elapsed_seconds=time.perf_counter()-arm_start,max_state_norm=max(tail),
                        counts=counts,budget=budget)
                    if per_entity and correct!=total:cell['status']='failed_positive_exact_control'
                results['cells'].append(cell)
                print(f"E{difficulty['entities']} seed{seed} {split}: global={cell['arms']['global']['accuracy']} perentity={cell['arms']['per_entity']['accuracy']} {cell['status']}",flush=True)
    results['preserved_p0_after']=p0_hashes()
    results['p0_bytes_unchanged']=before==results['preserved_p0_after']
    results['elapsed_seconds']=time.perf_counter()-start
    results['aggregate']={}
    for entities in (1,3):
        cells=[c for c in results['cells'] if c['difficulty']['entities']==entities and 'arms' in c]
        results['aggregate'][str(entities)]={arm:dict(correct=sum(c['arms'][arm]['correct'] for c in cells),
             queries=sum(c['arms'][arm]['queries'] for c in cells)) for arm in ('global','per_entity')}
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:results[k] for k in ('tests','aggregate','p0_bytes_unchanged','elapsed_seconds')}))
    return 0 if result.wasSuccessful() and results['p0_bytes_unchanged'] and all(c['status']=='executed' for c in results['cells']) else 1


if __name__=='__main__':sys.exit(main())
