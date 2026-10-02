"""Freeze exact handoff bytes after execution, or verify them without imports.

Running freeze again changes manifest identity; no historical results deleted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FILES=('operators.py','test_algebra.py','run_diagnostics.py','native_probe.py',
       'freeze_manifest.py','protocol.json','results.json','tests.log',
       'native_receipt.json','native_results.json','native_tests.log','HANDOFF.md')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    path=HERE/'manifest.json'
    if args.verify:
        m=json.loads(path.read_text())
        bad=[name for name,digest in m['files'].items() if not (HERE/name).is_file() or sha(HERE/name)!=digest]
        external_bad=[name for name,digest in m['read_only_dependencies'].items() if sha(ROOT/name)!=digest]
        if bad or external_bad: raise SystemExit(f'hash mismatch: {bad}, dependencies: {external_bad}')
        print('PASS frozen manifest: '+sha(path));return
    deps=('research/mamba3h/PROGRAM.md','research/mamba3h/integration/CONTRACT.md',
          'OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd',
          'OXN/nsos/tests/mamba3_validation/test_scan_recompute.py',
          'OXN/nsos/tests/mamba3_validation/mamba3_scan_oracle.py')
    m=dict(schema='mamba3h-algebra-p0-v1',baseline_commit='4db1250e7435318e1c2522b27649a5d5433f2f48',
           created_utc=datetime.now(timezone.utc).isoformat(),device='CPU',max_threads=2,
           python=platform.python_version(),scope='isolated algebra; native reference read-only; integration unattempted',
           files={name:sha(HERE/name) for name in FILES},
           read_only_dependencies={name:sha(ROOT/name) for name in deps})
    path.write_text(json.dumps(m,indent=2)+'\n')
    print('FROZEN manifest sha256 '+sha(path))


if __name__=='__main__':main()
