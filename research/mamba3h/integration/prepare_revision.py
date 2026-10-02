"""Declare causal wiring correction before rerun; preserve all v1 outcomes."""
from pathlib import Path
import hashlib,json,shutil,datetime
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/mamba3h'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=AREA/'review/pilot';new=AREA/'review/pilot-v2';protocol=json.loads((AREA/'manifests/pilot-v1.json').read_text())
    protocol.update(protocol='P0-pilot-v2-causal-wiring-correction',memory_input_M3='algebra_pre_read',
        predeclared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_grid='review/pilot',
        reuse_arms=['M0','M1','M2','MA','MC'],aggregate_learned_wall_seconds=900,
        new_jobs=5,new_job_wall_budget_seconds=90,parent_grid_wall_seconds=139.64336619980168,
        reason='M3 v1 memory consumed native z; v2 memory consumes algebra ast before retrieval. Explicit scientific wiring correction requested by coordinator; not accuracy-based hyperparameter selection.',
        hyperparameter_changes={},model='native features plus visible-input residual; M3 serial algebra pre-read state -> memory; other branches unchanged',
        claim_scope='structured auxiliary global state coupled to causal slots; frozen actual-native backbone, not native SSM modification')
    path=AREA/'manifests/pilot-v2.json'
    assert not path.exists(),'Do not overwrite prior declaration'
    path.write_text(json.dumps(protocol,indent=2)+'\n');cells=[];datasets=[]
    for seed in protocol['seeds']:
        src=old/f'seed-{seed}';dst=new/f'seed-{seed}';dst.mkdir(parents=True,exist_ok=True)
        for name in ('episodes.json','input.npz','native.npz','native.json','native.log'):
            shutil.copyfile(src/name,dst/name);assert sha(src/name)==sha(dst/name)
            datasets.append(dict(seed=seed,path=str((dst/name).relative_to(AREA)),sha256=sha(dst/name)))
        for arm in protocol['reuse_arms']:
            shutil.copyfile(src/f'{arm}.json',dst/f'{arm}.json')
            cells.append(dict(seed=seed,arm=arm,source=str((src/f'{arm}.json').relative_to(AREA)),sha256=sha(src/f'{arm}.json')))
    shutil.copyfile(old/'dataset-manifest.json',new/'dataset-manifest.json')
    (new/'reuse-manifest.json').write_text(json.dumps(dict(cells=cells,datasets=datasets,parent_summary_sha256=sha(old/'summary.json'),protocol_sha256=sha(path)),indent=2)+'\n')
    print('DECLARED v2; preserved v1, copied exact datasets and 25 unchanged cells; only five M3 jobs prepared')
if __name__=='__main__':main()
