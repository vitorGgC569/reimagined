"""Writes only p1_checks; python -B .../p1_checks/run_checks.py."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import sys
sys.dont_write_bytecode=True
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import unittest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from research.mamba3h.memory.p1_checks.checks import PositiveAndCausalTests,run_comparison,torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_status():
    parent=HERE.parent
    manifest=json.loads((parent/'manifest.json').read_text())
    hashes={name:sha(parent/name) for name in manifest['files_sha256']}
    assert hashes==manifest['files_sha256'],'frozen original changed'
    assert sha(parent/'manifest.json')=='376ba6fb332a77235950c9a3ad0fb41daa2f488b60c7360d60e76848417b6149'
    return {'manifest_sha256':sha(parent/'manifest.json'),'files_sha256':hashes}


def main():
    start=time.perf_counter()
    attempt=1
    while (HERE/f'attempt-{attempt}.json').exists():attempt+=1
    receipt={'scope':'isolated_no_update_P1_memory_checks','attempt':attempt,
             'protocol_sha256':sha(HERE/'protocol.json'),
             'source_sha256':{name:sha(HERE/name) for name in ('checks.py','routing.py','run_checks.py')},
             'frozen_before':frozen_status(),'optimizer_updates':0,
             'torch_version':torch.__version__,'intra_threads':torch.get_num_threads(),
             'interop_threads':torch.get_num_interop_threads(),
             'preimport_thread_env':{name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    stream=io.StringIO()
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PositiveAndCausalTests))
    receipt['tests']={'run':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'passed':tests.wasSuccessful()}
    receipt['failure_details']=[{'test':str(test),'traceback':detail} for test,detail in tests.failures+tests.errors]
    if tests.wasSuccessful():
        try:
            receipt['comparisons']=run_comparison()
        except Exception:
            receipt['comparison_error']=traceback.format_exc()
    receipt['status']='passed' if tests.wasSuccessful() and 'comparisons' in receipt else 'failed'
    receipt['frozen_after']=frozen_status()
    receipt['elapsed_seconds']=time.perf_counter()-start
    log=stream.getvalue()
    (HERE/f'attempt-{attempt}.log').write_text(log)
    (HERE/f'attempt-{attempt}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (HERE/'results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(log,flush=True)
    print(json.dumps({k:receipt[k] for k in ('status','tests','elapsed_seconds')},indent=2),flush=True)
    if 'comparisons' in receipt:
        for r in receipt['comparisons']:
            print(json.dumps({k:r[k] for k in ('seed','arm','query_candidate_count_per_sample','qk_nonzero_elements','state_bytes_per_sample','parameter_bytes_including_readout')},sort_keys=True))
    raise SystemExit(0 if receipt['status']=='passed' else 1)


if __name__=='__main__':main()
