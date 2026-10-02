"""Verify completed P1 peer files and copy immutable review inputs separately."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h'
OUT=AREA/'review/p1/handoffs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'receipt.json').exists(),'Completed handoff receipt is immutable'
receipts=[]
for folder,name,expected in (
    ('algebra/p1_checks','manifest.json','539e8ddb6988bd55a85acdc2b5200e43491fd3fca2719f691e1a7f852ac99076'),
    ('memory/p1_checks','manifest.json','e5f89a64ec35145bb95462ada9c9408f6c4d2099108cdfac496f77489f7ee5ae'),
    ('benchmarks/p1_positive','manifests/SHA256.json',None)):
    base=AREA/folder;mp=base/name;manifest=json.loads(mp.read_text())
    if expected:assert sha(mp)==expected
    files=manifest.get('files_sha256',manifest.get('files'))
    pairs=[(e['path'],e['sha256']) for e in files] if isinstance(files,list) else list(files.items())
    for path,h in pairs:
        p=base/path;assert p.resolve().is_relative_to(base.resolve());assert sha(p)==h
        dest=OUT/folder/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==h,'Partial handoff copy drift'
        else:shutil.copy2(p,dest)
        assert sha(dest)==h
    dest=OUT/folder/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==sha(mp)
    else:shutil.copy2(mp,dest)
    receipts.append(dict(folder=folder,manifest_path=name,manifest_sha256=sha(mp),files_verified=len(pairs),aggregate_sha256=manifest.get('aggregate_sha256')))
old=json.loads((AREA/'manifests/root2-manifest.json').read_text())
for e in old['files']:assert sha(AREA/e['path'])==e['sha256'],'Old Root2 receipt drift: '+e['path']
freeze=json.loads((AREA/'manifests/freeze.json').read_text());drift=[]
for e in freeze['files']:
    current=sha(ROOT/e['path'])
    if current!=e['sha256']:drift.append(dict(path=e['path'],frozen_sha256=e['sha256'],current_sha256=current))
assert all(e['path'].startswith('OXN/nsos/') for e in drift),'Optional frozen contract/program drift'
provider='OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd'
assert sha(AREA/'integration/p1/provider/nsos_ext.cp312-win_amd64.pyd')==next(e['sha256'] for e in freeze['files'] if e['path']==provider)
result=dict(peer_receipts=receipts,old_root2_files_verified=len(old['files']),old_manifest_sha256=sha(AREA/'manifests/root2-manifest.json'),
    shared_baseline_drift=drift,private_exact_provider_verified=True,no_original_files_edited=True)
(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
