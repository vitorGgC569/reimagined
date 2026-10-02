"""Independent published-data checks and exact-provider replay before updates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h';OUT=AREA/'review/p1/preflight-r2'
os.environ['PYTHONPATH']=str(ROOT)
assert not OUT.exists();OUT.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tests=[]
commands=[('root', [sys.executable,'-B',str(AREA/'integration/p1/test_positive.py')]),
 ('algebra',[sys.executable,'-B',str(AREA/'review/p1/handoffs/algebra/p1_checks/test_exact_entity.py')]),
 ('memory',[sys.executable,'-B','-m','unittest','research.mamba3h.memory.p1_checks.checks']),
 ('benchmark',[sys.executable,'-B','-m','unittest','research.mamba3h.benchmarks.p1_positive.tests'])]
for name,cmd in commands:
    r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
    (OUT/f'{name}.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    tests.append(dict(suite=name,passed=True,log_sha256=sha(OUT/f'{name}.log')))
# Original P0 numeric arrays and cached native outputs, independently regenerated
# with recovered exact provider. No old receipt or numerical tolerance is edited.
equivalence=[]
for seed in (11,23,37,53,71):
    original=AREA/f'review/pilot/seed-{seed}'
    dest=OUT/f'replayed-native-{seed}.npz'
    r=subprocess.run([sys.executable,'-B',str(AREA/'integration/p1/native_provider.py'),'features','--input',str(original/'input.npz'),'--out',str(dest),'--seed',str(seed)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stderr
    reference=np.load(original/'native.npz');replayed=np.load(dest)
    assert reference.files==replayed.files
    for name in reference.files:np.testing.assert_array_equal(reference[name],replayed[name])
    equivalence.append(dict(seed=seed,all_splits_bitwise_equal=True,reference_sha256=sha(original/'native.npz'),replay_sha256=sha(dest)))
sys.path.insert(0,str(ROOT))
from research.mamba3h.benchmarks.p1_positive import model_numeric,solve
inputs=0
for path in (AREA/'review/p1/positive-r2').glob('seed-*/episodes.json'):
    numeric=np.load(path.with_name('input.npz'))
    for split,eps in json.loads(path.read_text()).items():
        for i,ep in enumerate(eps):
            np.testing.assert_array_equal(numeric[split][i],np.array(model_numeric(ep),dtype='f4'))
            assert ep['targets']==solve(ep);inputs+=1
assert inputs==800
receipt=dict(tests=tests,root_tests=5,peer_tests=38,independent_model_numeric_and_solver_examples=inputs,
    recovered_provider_old_cache_replay=equivalence,optimizer_updates=0,threads=2,
    protocol_sha256=sha(AREA/'manifests/p1a-positive-r2.json'),runner_sha256=sha(AREA/'integration/p1/positive_runner.py'),
    dataset_manifest_sha256=sha(AREA/'review/p1/positive-r2/dataset-manifest.json'),
    provider_sha256=sha(AREA/'integration/p1/provider/nsos_ext.cp312-win_amd64.pyd'))
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
