"""Copy explicitly released peer APIs into Root2-owned immutable integration."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'research/mamba3h/integration/vendor'
FILES=['algebra/operators.py','memory/slots.py','memory/__init__.py',
       'benchmarks/__init__.py','benchmarks/schema.py','benchmarks/generation.py','benchmarks/oracle.py','benchmarks/protocol.py']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    manifest=[]
    for path in FILES:
        src=ROOT/'research/mamba3h'/path;raw=src.read_bytes()
        dst=DEST/'research/mamba3h'/path;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert dst.read_bytes()==raw,'Do not silently overwrite snapshot: '+path
        else:dst.write_bytes(raw)
        assert sha(src.read_bytes())==sha(raw),'Peer changed during handoff capture'
        manifest.append(dict(owner=path.split('/')[0],source=str(src.relative_to(ROOT)),sha256=sha(raw),bytes=len(raw)))
    for path in ('research/__init__.py','research/mamba3h/__init__.py','research/mamba3h/algebra/__init__.py'):
        dst=DEST/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text('"""Root2 immutable handoff namespace."""\n')
    (DEST/'snapshot.json').write_text(json.dumps(dict(label='released_peer_API_snapshot',files=manifest),indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
