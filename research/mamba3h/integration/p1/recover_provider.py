"""Preserve setup failure, privately recover exact provider, declare P1 revision."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
out=AREA/'review/p1/positive';history=out/'setup-failure-preserved'
assert not history.exists();history.mkdir()
for src in (AREA/'integration/p1/positive_runner.py',AREA/'manifests/p1a-positive.json'):
    shutil.copy2(src,history/src.name)
protocol=json.loads((AREA/'manifests/p1a-positive.json').read_text())
ledger=json.loads((out/'run-ledger.json').read_text())
ledger.append(dict(seed=23,arm='M0',status='failed_setup',optimizer_updates=0,
    reason='child attempted without prepared seed; parent failed writing missing seed directory log',
    measured_wall_seconds=None,conservative_budget_charge_seconds=90))
seen={(j['seed'],j['arm']) for j in ledger}
for s in protocol['seeds']:
    for a in protocol['arms']:
        if (s,a) not in seen:ledger.append(dict(seed=s,arm=a,status='unattempted',optimizer_updates=0))
charge=sum(j.get('wall_seconds',0)+j.get('conservative_budget_charge_seconds',0) for j in ledger)
source=ROOT/'artifacts/maestri_integral_20261001_gpuopt/root-baseline-4db1250/OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd'
freeze=json.loads((AREA/'manifests/freeze.json').read_text())
expected=next(e['sha256'] for e in freeze['files'] if e['path'].endswith('.pyd') and 'build-gm-cpu' in e['path'])
assert sha(source)==expected
dst=AREA/'integration/p1/provider'/source.name;dst.parent.mkdir(parents=True,exist_ok=True)
assert not dst.exists();shutil.copy2(source,dst);assert sha(dst)==expected
put(history/'failure-audit.json',dict(prepare_status='failed_frozen_provider_guard',optimizer_updates=0,
    launch_error='PowerShell semicolon ran run despite failed preparation; run lacked readiness guard',
    preserved_ledger=ledger,aggregate_conservative_charge_seconds=charge,
    shared_current_provider_sha256=sha(ROOT/'OXN/nsos/build-gm-cpu'/source.name),
    recovered_sha256=expected,recovered_from=str(source.relative_to(ROOT)),private_path=str(dst.relative_to(ROOT))))
protocol.update(protocol='P1a-native-frozen-positive-addressing-r2',
    setup_revision='exact SHA provider relocated privately; readiness and no-overwrite guard; data/weights/hparams unchanged; zero previous updates',
    provider_path=str(dst.relative_to(ROOT)),provider_sha256=expected,
    original_setup_failures=6,original_setup_unattempted=19,
    setup_failure_conservative_charge_seconds=charge,
    new_learned_wall_budget_seconds=protocol['new_learned_wall_budget_seconds']-charge,
    prior_root2_learned_wall_seconds=protocol['prior_root2_learned_wall_seconds']+charge)
put(AREA/'manifests/p1a-positive-r2.json',protocol)
print(json.dumps(dict(private_provider_sha256=sha(dst),remaining_p1_budget_seconds=protocol['new_learned_wall_budget_seconds'],setup_charge_seconds=charge)))
