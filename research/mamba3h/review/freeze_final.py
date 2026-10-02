"""SHA-pin Root2 optional evidence and verify it without rewriting receipts."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h';OUT=AREA/'manifests/root2-manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    files=[]
    for folder in ('integration','review','manifests'):
        for p in sorted((AREA/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p==OUT:continue
            # Async Maestri outputs are transport logs, not scientific receipts.
            if p.name.startswith(('dispatch-','clarify-','progress-','pilot-Lyra','pilot-Neris','pilot-Orin','mid-Codex','wiring-Codex','finish-Orin','handoff-')):continue
            files.append(dict(path=p.relative_to(AREA).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    return files
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','verify']);a=p.parse_args()
    baseline=json.loads((AREA/'manifests/freeze.json').read_text());drift=[]
    for e in baseline['files']:
        if sha(ROOT/e['path'])!=e['sha256']:drift.append(e['path'])
    assert not drift,'Frozen baseline drift: '+str(drift)
    if a.mode=='freeze':
        assert not OUT.exists(),'Final evidence already frozen; use verify'
        info=dict(schema='root2-mamba3h-p0-final-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            baseline_commit=baseline['baseline_commit'],observed_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            production_diff=subprocess.check_output(['git','diff','--','OXN/nsos'],cwd=ROOT,text=True),
            scope='Root2 integration/review/manifests only; completed copied handoffs',files=inventory(),
            claim_scope='actual native frozen backbone plus optional adapters; integrated advantage not demonstrated',
            native_end_to_end='unattempted',GPU=False,commit_push=False,threads_max=2)
        assert not info['production_diff'];OUT.write_text(json.dumps(info,indent=2)+'\n')
    info=json.loads(OUT.read_text());failed=[]
    for e in info['files']:
        path=AREA/e['path'];assert path.resolve().is_relative_to(AREA.resolve())
        if not path.exists() or path.stat().st_size!=e['bytes'] or sha(path)!=e['sha256']:failed.append(e['path'])
    assert not failed,'Final evidence drift: '+str(failed)
    print(json.dumps(dict(passed=True,files_checked=len(info['files']),baseline_files_checked=len(baseline['files']),
        manifest_sha256=sha(OUT),baseline_commit=info['baseline_commit'],GPU=False)))
if __name__=='__main__':main()
