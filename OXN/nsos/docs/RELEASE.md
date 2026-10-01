# NSOS Product Release Checklist

This checklist applies to the supported product surface: `OXN/nsos`.

## Required Local Gate

Run from the repository root:

```powershell
cmake -S .\OXN\nsos -B .\OXN\nsos\build-mvp `
  -DNSOS_GPU_BACKEND=NONE `
  -DNSOS_BUILD_PYTHON=ON `
  -DNSOS_BUILD_TESTS=ON `
  -DNSOS_BUILD_CLI=ON `
  -DNSOS_BUILD_API=ON `
  -DNSOS_BUILD_OXTAMEM=OFF

cmake --build .\OXN\nsos\build-mvp --config Release
ctest --test-dir .\OXN\nsos\build-mvp -C Release --output-on-failure

$env:NSOS_BUILD_DIR = (Resolve-Path .\OXN\nsos\build-mvp).Path
python .\OXN\scripts\gatekeeper.py

python .\OXN\nsos\scripts\benchmark_gate.py `
  --build-dir .\OXN\nsos\build-mvp `
  --device cpu `
  --profile mamba_small `
  --max-tokens 8 `
  --report-path .\OXN\nsos\build-mvp\benchmark_gate_cpu.local.json

python .\OXN\nsos\scripts\fuzz_surface_smoke.py `
  --build-dir .\OXN\nsos\build-mvp `
  --report-path .\OXN\nsos\build-mvp\fuzz_surface_smoke.local.json
```

## Integrated Module Gate

```powershell
cargo test --manifest-path .\modules\oxtamem\oxta_engine\Cargo.toml --all-targets
cargo clippy --manifest-path .\modules\oxtamem\oxta_engine\Cargo.toml --all-targets -- -D warnings
python -m compileall -q .\modules\oxtamem\python .\modules\oxtamem\nn
```

## Docker Gate

```powershell
docker build -t nsos-mvp-local .
```

## Release Definition

A product release is ready only when:

- clean CPU build succeeds
- product CTest passes
- `test_layer_audit` passes as part of CTest
- gatekeeper passes
- benchmark gate passes
- fuzz smoke passes
- OxtaMem integrated lane passes
- Docker image builds
- `git status --short` is clean except ignored local build artifacts
- no build/cache/checkpoint/log artifacts are staged

GPU remains experimental until the real-hardware and backend-appropriate
memory-checker lanes in `PRODUCT.md` are closed. CPU/Rust release gates do not
certify either HIP or CUDA. Opt-in training redesign results and unresolved
limits are recorded in `GPU_TRAINING_REDESIGN_2026-09-29.md`.
