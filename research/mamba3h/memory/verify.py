"""Capture executable correctness receipt and hash the owned handoff files."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="2"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','freeze'))
    args=parser.parse_args()
    if args.mode=='test':
        start=time.perf_counter()
        cmd=[sys.executable,'-m','unittest','research.mamba3h.memory.test_slots','-v']
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=30)
        (HERE/'test_results.log').write_text(result.stdout+result.stderr)
        receipt={'command':cmd,'returncode':result.returncode,'elapsed_seconds':time.perf_counter()-start,
                 'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('slots.py','test_slots.py','__init__.py')},
                 'thread_env':{name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                 'scope':'isolated_component_CPU_correctness_not_native_integration'}
        (HERE/'test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(result.stdout+result.stderr)
        raise SystemExit(result.returncode)
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.iterdir())
            if path.is_file() and path.name!='manifest.json'}
    manifest={'baseline_read_only':'4db1250e7435318e1c2522b27649a5d5433f2f48',
              'scope':'research/mamba3h/memory_only','files_sha256':hashes}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
