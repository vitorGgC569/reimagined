# NSOS MVP Workspace

This repository contains several research and legacy tracks. The supported MVP
surface is `OXN/nsos`.

## MVP Scope

Stable for MVP:

- CPU inference through `nsos_api_server`, `nsos_cli`, and `nsos_ext`.
- Model pack load/save with manifest checksums and bounded pack parsing.
- HTTP API with authentication required by default.
- Local development mode only when `--allow-unauthenticated-local` is explicit.

Experimental:

- CUDA/GPU execution and `test_gpu_parity`.
- OxtaMem as a network service.
- Distributed/MPI orchestration.
- TTT product flows.
- Lean/formal self-healing claims.

Legacy or research-only:

- Root-level `serve_oxn.py`.
- Old demo servers and mock scripts outside `OXN/nsos`.
- `KernelOpen`, `CHRASS`, `CART`, `OXB`, and hardware experiments unless they
  pass the same build/test gate.

## Build MVP

Windows example:

```powershell
cmake -S .\OXN\nsos -B .\OXN\nsos\build-mvp `
  -DNSOS_ENABLE_CUDA=OFF `
  -DNSOS_BUILD_PYTHON=ON `
  -DNSOS_BUILD_TESTS=ON `
  -DNSOS_BUILD_CLI=ON `
  -DNSOS_BUILD_API=ON `
  -DNSOS_BUILD_OXTAMEM=OFF

cmake --build .\OXN\nsos\build-mvp --config Release
ctest --test-dir .\OXN\nsos\build-mvp -C Release --output-on-failure
```

## Run API

```powershell
$env:NSOS_API_TOKEN = "change-me"
.\OXN\nsos\build-mvp\Release\nsos_api_server.exe `
  --host 127.0.0.1 `
  --port 8080 `
  --auth-token $env:NSOS_API_TOKEN
```

Admin endpoints such as `/train-*` and `/pack` are disabled unless the server is
started with `--enable-admin-endpoints`.

## Docker

```bash
docker build -t nsos-mvp .
docker run --rm -p 8080:8080 -e NSOS_API_TOKEN=change-me nsos-mvp
```

## Audit

See `RELATORIO_AUDITORIA_MVP_2026-04-26.md` for the full audit that drove the
MVP hardening work.
