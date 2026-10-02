"""Read-only native CPU differential gate, reusing pinned existing validation.

This validates actual-native Mamba3 versus its independent NumPy oracle.
It does not integrate the new algebra adapter or validate a new recurrence port.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASELINE='4db1250e7435318e1c2522b27649a5d5433f2f48'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provider=ROOT/'OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd'
    test=ROOT/'OXN/nsos/tests/mamba3_validation/test_scan_recompute.py'
    oracle=test.with_name('mamba3_scan_oracle.py')
    record=dict(scope='native_CPU_reference_gate_NOT_algebra_integration',gpu_executed=False,
                baseline_commit=BASELINE,provider=str(provider),timeout_seconds=90,
                source_test_sha256=sha(test),source_oracle_sha256=sha(oracle))
    if not provider.is_file():
        record.update(status='unattempted_missing_native_provider')
    else:
        record['provider_sha256_before']=sha(provider)
        env=dict(os.environ)
        env.update({key:'2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        env['PYTHONDONTWRITEBYTECODE']='1'
        cmd=[sys.executable,str(test),'--native-module-dir',str(provider.parent),
             '--result',str(HERE/'native_results.json')]
        started=time.perf_counter()
        with (HERE/'native_tests.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                record.update(status='executed' if result.returncode==0 else 'failed_native_gate',returncode=result.returncode)
            except subprocess.TimeoutExpired:
                record.update(status='censored_native_timeout')
        record['elapsed_seconds']=time.perf_counter()-started
        record['provider_sha256_after']=sha(provider)
        record['provider_unchanged']=record['provider_sha256_before']==record['provider_sha256_after']
    (HERE/'native_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return 0 if record['status']=='executed' and record.get('provider_unchanged') else 1


if __name__=='__main__':sys.exit(main())
