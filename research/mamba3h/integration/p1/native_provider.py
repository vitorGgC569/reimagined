"""Location-only bridge to exact P0 code and SHA-pinned private native binary.

Never accepts a replacement binary by semantic similarity. Original P0 source,
provider guard, numerical tolerances and feature generation stay intact.
"""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
AREA=ROOT/'research/mamba3h'
original=AREA/'integration/native_cpu.py'
manifest=json.loads((AREA/'manifests/root2-manifest.json').read_text())
expected=next(e['sha256'] for e in manifest['files'] if e['path']=='integration/native_cpu.py')
assert hashlib.sha256(original.read_bytes()).hexdigest()==expected,'Frozen P0 bridge source drift'
source=original.read_text()
old="BUILD=ROOT/'OXN/nsos/build-gm-cpu'"
new="BUILD=ROOT/'research/mamba3h/integration/p1/provider'"
assert source.count(old)==1
# The original guard subsequently checks this private provider against freeze.json.
exec(compile(source.replace(old,new),str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
