"""Freeze identities; does not assert that binary was built from this commit."""
from pathlib import Path
import hashlib,json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[3]
PIN='4db1250e7435318e1c2522b27649a5d5433f2f48'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert head==PIN
    paths=['research/mamba3h/PROGRAM.md','research/mamba3h/integration/CONTRACT.md',
        'OXN/nsos/src/mamba3_layer.cpp','OXN/nsos/include/mamba3_layer_math.h','OXN/nsos/src/bindings.cpp',
        'OXN/nsos/build-gm-cpu/nsos_ext.cp312-win_amd64.pyd','OXN/nsos/build-gm-cpu/test_mamba3_layer.exe',
        'OXN/nsos/build-gm-cpu/test_mamba3_integral_reference.exe','OXN/nsos/build-gm-cpu/mamba3_math_oracle_probe.exe']
    info=dict(baseline_commit=PIN,observed_head=head,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in paths],
        inherited_HIP_extension_sha256='8212a02b9e1486e84352aaeba2c42b02072d3b775d2f4cc2b6b98f5dbb8f5060',
        native_binary_commit_provenance='not independently certified by hash; equivalence tested separately',
        production_diff=subprocess.check_output(['git','diff','--','OXN/nsos'],cwd=ROOT,text=True),
        max_threads_per_process=2,gpu_executed=False)
    dst=ROOT/'research/mamba3h/manifests/freeze.json'
    if dst.exists():
        old=json.loads(dst.read_text());assert old['files']==info['files'],'freeze drift: do not overwrite'
    else: dst.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
