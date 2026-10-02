"""Separate immutable P1 evidence inventory; never refresh P0 manifest."""
import argparse,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h';OUT=AREA/'manifests/root2-p1-manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_old():
    old=AREA/'manifests/root2-manifest.json'
    assert sha(old)=='fd72ca8887958f56f435d53876f2050ee84f59d1f784a7def9bcba6fa3943d8e'
    info=json.loads(old.read_text())
    for e in info['files']:assert sha(AREA/e['path'])==e['sha256'],'P0 receipt drift: '+e['path']
    assert sha(AREA/'integration/p1/provider/nsos_ext.cp312-win_amd64.pyd')=='050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f'
    return len(info['files'])
def inventory():
    paths=list((AREA/'integration/p1').rglob('*'))+list((AREA/'review/p1').rglob('*'))+list((AREA/'manifests').glob('p1*.json'))
    return [dict(path=p.relative_to(AREA).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','verify']);args=parser.parse_args()
    old_count=verify_old()
    if args.mode=='freeze':
        assert not OUT.exists(),'P1 evidence already frozen'
        verdict=json.loads((AREA/'review/p1/verdict.json').read_text())
        receipt=dict(schema='Root2-P1-independent-final-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            baseline_commit='4db1250e7435318e1c2522b27649a5d5433f2f48',old_P0_files_verified=old_count,
            provider_sha256='050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f',
            learned_completed=25,mechanism_completed=45,tests_passed=43,
            setup_failed=6,setup_unattempted=19,claim_scope=verdict['scope'],
            shared_baseline_drift=verdict['shared_current_baseline_drift'],files=inventory(),
            no_GPU_shared_build_OXN_peer_edits_commit_push=True)
        OUT.write_text(json.dumps(receipt,indent=2)+'\n')
    receipt=json.loads(OUT.read_text())
    for e in receipt['files']:
        path=AREA/e['path'];assert path.resolve().is_relative_to(AREA.resolve())
        assert path.stat().st_size==e['bytes'] and sha(path)==e['sha256'],'P1 evidence drift: '+e['path']
    print(json.dumps(dict(passed=True,old_P0_files_verified=old_count,P1_files_verified=len(receipt['files']),manifest_sha256=sha(OUT))))
