"""Independent rerun on completed copied sources; preserve original receipts."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h';SNAP=AREA/'review/handoffs'
sys.dont_write_bytecode=True;sys.path.insert(0,str(SNAP));sys.path.insert(0,str(SNAP/'research/mamba3h/algebra'))
from research.mamba3h.benchmarks import compact_targets,encode_numeric
from research.mamba3h.benchmarks.protocol import gap_recovery

def main():
    # Each peer test owns process-local torch thread configuration. Combining
    # them in one interpreter would attempt to re-set interop after work starts.
    start=time.perf_counter();runs=[]
    for name,expected in [('research.mamba3h.algebra.test_algebra',9),('research.mamba3h.memory.test_slots',20),('research.mamba3h.benchmarks.tests',50)]:
        code=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(SNAP)!r});sys.path.insert(0,{str(SNAP/'research/mamba3h/algebra')!r});import unittest;s=unittest.defaultTestLoader.loadTestsFromName({name!r});r=unittest.TextTestRunner(verbosity=2).run(s);assert r.testsRun=={expected};sys.exit(0 if r.wasSuccessful() else 1)"
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
        (AREA/'review'/f'final-{name.split(".")[-2]}-tests.log').write_text(r.stdout+r.stderr)
        runs.append(dict(module=name,expected_tests=expected,returncode=r.returncode));assert r.returncode==0,r.stderr
    assert not gap_recovery(.999,.99,1.)['defined'],'threshold boundary must remain undefined'
    # All 800 pilot episodes must equal Orin final pinned corpus, encodings and
    # compact target mapping used by the executed frozen API version.
    compared=0
    for seed in (11,23,37,53,71):
        parent=AREA/'review/pilot'/f'seed-{seed}';eps=json.loads((parent/'episodes.json').read_text())
        for split,episodes in eps.items():
            path=SNAP/'research/mamba3h/benchmarks/datasets/pilot-v1/inst'/f'seed-{seed}'/f'{split}.jsonl'
            owned=[json.loads(line) for line in path.read_text().splitlines()]
            assert episodes==owned,'Prepared peer corpus differs from executed data'
            for ep in episodes:
                assert compact_targets(ep)==[5 if t==125 else t for t in ep['targets']]
                compared+=1
    info=dict(passed=True,unit_tests=sum(r['expected_tests'] for r in runs),failures=0,errors=0,runs=runs,
        elapsed_seconds=time.perf_counter()-start,pilot_episodes_verified=compared,physical_native_empty_supported=False,
        copied_completed_sources=True,threads=2,gpu_executed=False)
    receipt=AREA/'review/final-peer-tests.json'
    if not receipt.exists():receipt.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))
if __name__=='__main__':main()
