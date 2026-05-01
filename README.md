# NSOS MVP Workspace

This repository is intentionally a monorepo, but it is not a single release
surface. The boundary is tracked in `PROJECT_BOUNDARY.json` and explained in
`docs/PROJECT_BOUNDARY.md`.

## Repository Boundary

Supported product:

- `OXN/nsos`

Integrated companion module:

- `modules/oxtamem`

Research/incubation by default:

- everything else unless promoted through the project boundary process

Incubation code can be useful and actively developed, but product claims,
release blockers, and compatibility promises belong only to the supported
product and integrated-module surfaces.

## Product Scope

Stable for the current NSOS MVP:

- CPU inference through `nsos_api_server`, `nsos_cli`, and `nsos_ext`.
- Model pack load/save with manifest checksums and bounded pack parsing.
- HTTP API with authentication required by default.
- Local development mode only when `--allow-unauthenticated-local` is explicit.
- OxtaMem integration when its Rust/Python gates pass.

Not release-supported yet:

- CUDA/GPU execution beyond explicitly gated lanes.
- Distributed/MPI orchestration.
- TTT product workflows.
- Lean/formal self-healing claims.
- `KernelOpen`, `CHRASS`, `CART`, `OXB`, root-level Pantheon code, hardware
  experiments, root demos, and old benchmark/test tracks until promoted.

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
