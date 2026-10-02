"""Separate P1 freeze; never mutates P0. Freeze is write-once."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    target=HERE/'manifest.json'
    if args.verify:
        manifest=json.loads(target.read_text())
        bad=[name for name,digest in manifest['files'].items() if sha(HERE/name)!=digest]
        bad.extend(name for name,digest in manifest['read_only_dependencies'].items() if sha(ROOT/name)!=digest)
        if bad:raise SystemExit('SHA mismatch '+repr(bad))
        print('PASS P1 manifest SHA256 '+sha(target));return
    if target.exists():raise SystemExit('P1 manifest already frozen; use --verify')
    names=('exact_entity.py','test_exact_entity.py','run_checks.py','protocol.json','HANDOFF.md','freeze.py','results.json','tests.log')
    deps=('research/mamba3h/benchmarks/oracle.py','research/mamba3h/benchmarks/schema.py',
          'research/mamba3h/benchmarks/generation.py','research/mamba3h/benchmarks/__init__.py',
          'research/mamba3h/algebra/HANDOFF.md','research/mamba3h/algebra/manifest.json')
    manifest=dict(schema='algebra-p1-exact-known-control-v1',scope='isolated_exact_known_operator_control_NOT_actualnative_NCE2E',
                  created_utc=datetime.now(timezone.utc).isoformat(),device='CPU',max_threads=2,
                  learned=False,optimizer_updates=0,files={name:sha(HERE/name) for name in names},
                  read_only_dependencies={name:sha(ROOT/name) for name in deps})
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print('FROZEN P1 manifest SHA256 '+sha(target))


if __name__=='__main__':main()
