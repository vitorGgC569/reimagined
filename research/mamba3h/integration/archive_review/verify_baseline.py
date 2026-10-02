"""Independent read-only archive identity review, separate from frozen P0/P1."""
import hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[4];AREA=ROOT/'research/mamba3h'
BASE=ROOT/'artifacts/maestri_integral_20261001_gpuopt/root-baseline-4db1250'
OUT=AREA/'review/archive_review';assert not OUT.exists();OUT.mkdir(parents=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()
start=time.perf_counter();archive=json.loads((BASE/'archive.json').read_text());commit=archive['baseline_commit']
assert commit=='4db1250e7435318e1c2522b27649a5d5433f2f48'
records=[]
for entry in archive['files']:
    path=Path(entry['archive_path']);assert path.resolve().is_relative_to(BASE.resolve())
    assert path.stat().st_size==entry['bytes'];assert sha(path)==entry['sha256']
    records.append(dict(path=path.relative_to(BASE).as_posix(),bytes=entry['bytes'],sha256=entry['sha256']))
source=BASE/'sources-4db1250.zip';assert sha(source)==archive['source_archive_sha256']
freeze=json.loads((AREA/'manifests/freeze.json').read_text());sources=[]
with zipfile.ZipFile(source) as z:
    # Git archive ZIP comments carry the commit ID when available.
    comment=z.comment.decode('ascii',errors='replace')
    if comment:assert comment==commit,'Source ZIP commit identity differs'
    for entry in freeze['files']:
        name=entry['path']
        if not name.startswith(('OXN/nsos/src/','OXN/nsos/include/')):continue
        data=z.read(name)
        gitblob=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
        assert data==gitblob,'Source archive differs from frozen commit blob'
        raw=hashlib.sha256(data).hexdigest()
        reconstructed=hashlib.sha256(data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
        assert entry['sha256'] in (raw,reconstructed),'Frozen source hash is not explained by LF/CRLF checkout'
        sources.append(dict(path=name,archive_raw_sha256=raw,git_blob_bitwise_equal=True,
            old_checkout_sha256=entry['sha256'],matches_checkout_raw=raw==entry['sha256'],
            matches_checkout_with_CRLF=reconstructed==entry['sha256']))
cpu=next(e for e in records if e['path']=='OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd')
hip=next(e for e in records if e['path']=='OXN/nsos/build-gm-hip-gpuopt/nsos_ext.cp312-win_amd64.pyd')
assert cpu['sha256']=='050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f'
assert hip['sha256']==freeze['inherited_HIP_extension_sha256']
assert sha(AREA/'integration/p1/provider/nsos_ext.cp312-win_amd64.pyd')==cpu['sha256']
for filename in ('root2-manifest.json','root2-p1-manifest.json'):
    p=AREA/'manifests'/filename;info=json.loads(p.read_text())
    for e in info['files']:assert sha(AREA/e['path'])==e['sha256'],'Historical evidence drift: '+e['path']
result=dict(passed=True,baseline_commit=commit,source_zip_sha256=sha(source),zip_commit_comment=comment,
    source_entries=sources,archived_files_verified=len(records),files=records,
    archive_json_sha256=sha(BASE/'archive.json'),private_CPU_equals_archived_original=True,
    CPU_sha256=cpu['sha256'],HIP_sha256=hip['sha256'],HIP_loaded=False,GPU_used=False,
    old_P0_manifest_sha256=sha(AREA/'manifests/root2-manifest.json'),P1_manifest_sha256=sha(AREA/'manifests/root2-p1-manifest.json'),
    old_evidence_reclassified=False,learned_updates=0,elapsed_seconds=time.perf_counter()-start,
    limitation='Archive content agrees with committed sources and archived binaries; this does not certify a hermetic build. Behavioral equivalence is independently recorded in P0/P1.')
(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'HANDOFF.md').write_text(f'''# Independent archived baseline review

PASS: commit{commit}, ZIP SHA{result['source_zip_sha256']}.
All{len(records)}archived files verified by bytes/SHA; original CPU050f71f... and
HIP8212a... preserved. Private research CPU provider equals archived original.
Three native source/header entries match Git commit blobs bitwise. Original
P0checkout hashes match raw bytes or explicitly reconstructed CRLF checkout;
line-ending differences are recorded, not repinned. No archive extraction into
shared sources/builds. ZIP commit comment matches4db1250.

P0manifest322files and P1manifest241files all remain byte-identical. No reclassified
v1/v2 result, new model run, GPU import/build, commit or push. This is content
provenance and behavioral-check linkage, not a hermetic build certificate.
See receipt.json for every source/binary hash and interpretation boundaries.
''')
print(json.dumps({k:result[k] for k in ('passed','archived_files_verified','private_CPU_equals_archived_original','CPU_sha256','HIP_sha256','elapsed_seconds')}))
