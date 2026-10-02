"""Verify owner freezes and copy only completed handoffs, never mutate peers."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h';DST=AREA/'review/handoffs'
PINS={'algebra':'7a389aac8f7d54fef77344b6b8c9ef1704235586b40799d56e2f58a1c80ed43d',
      'memory':'376ba6fb332a77235950c9a3ad0fb41daa2f488b60c7360d60e76848417b6149'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    owners=[]
    for owner in ('algebra','memory','benchmarks'):
        base=AREA/owner;assert (base/'HANDOFF.md').exists(),'Handoff incomplete: '+owner
        manifest=base/('manifests/SHA256.json' if owner=='benchmarks' else 'manifest.json')
        raw=manifest.read_bytes();hash_manifest=sha(raw)
        if owner in PINS:assert hash_manifest==PINS[owner],'Owner manifest identity changed'
        m=json.loads(raw)
        files=m.get('files_sha256',m.get('files'))
        entries=[dict(path=k,sha256=v) for k,v in files.items()] if isinstance(files,dict) else files
        captured=[]
        for e in entries:
            src=base/e['path'];assert src.resolve().is_relative_to(base.resolve())
            data=src.read_bytes();assert sha(data)==e['sha256'],'Owner freeze drift: '+str(src)
            dst=DST/'research/mamba3h'/owner/e['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():assert dst.read_bytes()==data,'Do not overwrite immutable final handoff'
            else:dst.write_bytes(data)
            assert sha(src.read_bytes())==e['sha256'],'Concurrent owner change'
            captured.append(dict(path=e['path'],sha256=e['sha256'],bytes=len(data)))
        mdst=DST/'research/mamba3h'/owner/manifest.relative_to(base);mdst.parent.mkdir(parents=True,exist_ok=True);mdst.write_bytes(raw)
        owners.append(dict(owner=owner,manifest_sha256=hash_manifest,files_checked=len(captured),files=captured))
    for name in ('research/__init__.py','research/mamba3h/__init__.py','research/mamba3h/algebra/__init__.py'):
        p=DST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('"""Completed independently verified handoffs."""\n')
    (DST/'receipt.json').write_text(json.dumps(dict(owners=owners,source_mutation=False),indent=2)+'\n')
    print(json.dumps([dict(owner=o['owner'],files_checked=o['files_checked'],manifest_sha256=o['manifest_sha256']) for o in owners],indent=2))
if __name__=='__main__':main()
