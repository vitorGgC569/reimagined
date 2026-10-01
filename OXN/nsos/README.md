# NSOS MVP

NSOS is the supported neural runtime in this repository. It provides an
authenticated inference service, CLI and Python bindings, with CPU plus
selectable AMD HIP and NVIDIA CUDA backends.

## Supported In MVP

- `nsos_core` CPU runtime.
- `nsos_api_server` HTTP API.
- `nsos_cli` command-line companion.
- `nsos_ext` Python binding.
- Model pack load/save with bounded parsing, manifest version and checksums.
- Tokenizer pack load/save with bounded parsing.
- Native CTest suite plus Python binding smoke test.
- AMD HIP/ROCm GPU training validated on Radeon RX 7600 (`gfx1102`).
- NVIDIA CUDA backend retained with the same shared `.cu` kernel sources.

## Experimental Or Out Of Scope

- A GPU/driver combination is production-eligible only after its complete
  parity and end-to-end training gate passes on the target hardware.
- NVIDIA CUDA was not executed on the AMD validation workstation.
- Distributed/MPI orchestration is not an MVP production feature.
- TTT product flows are research-only unless a release profile explicitly tests
  them.
- OxtaMem network service is not a public service in the MVP. Keep it local or
  authenticate it.
- Lean/formal self-healing is not a production guarantee.

## Build

Windows CPU MVP:

```powershell
cmake -S .\OXN\nsos -B .\OXN\nsos\build-mvp `
  -DNSOS_GPU_BACKEND=NONE `
  -DNSOS_BUILD_PYTHON=ON `
  -DNSOS_BUILD_TESTS=ON `
  -DNSOS_BUILD_CLI=ON `
  -DNSOS_BUILD_API=ON `
  -DNSOS_BUILD_OXTAMEM=OFF `
  -DNSOS_ENABLE_TURBOQUANT=ON

cmake --build .\OXN\nsos\build-mvp --config Release
ctest --test-dir .\OXN\nsos\build-mvp -C Release --output-on-failure
```

Linux CPU MVP:

```bash
cmake -S OXN/nsos -B OXN/nsos/build-mvp \
  -DCMAKE_BUILD_TYPE=Release \
  -DNSOS_GPU_BACKEND=NONE \
  -DNSOS_BUILD_PYTHON=ON \
  -DNSOS_BUILD_TESTS=ON \
  -DNSOS_BUILD_CLI=ON \
  -DNSOS_BUILD_API=ON \
  -DNSOS_BUILD_OXTAMEM=OFF \
  -G Ninja

cmake --build OXN/nsos/build-mvp -j"$(nproc)"
ctest --test-dir OXN/nsos/build-mvp --output-on-failure
```

## HTTP API

Authentication is required by default. `/health` and `/ready` may remain
anonymous unless `--health-auth` is set.

```powershell
$env:NSOS_API_TOKEN = "change-me"
.\OXN\nsos\build-mvp\Release\nsos_api_server.exe `
  --model C:\path\to\model_pack `
  --host 127.0.0.1 `
  --port 8080 `
  --auth-token $env:NSOS_API_TOKEN
```

For local-only development without a token, the opt-in flag is explicit:

```powershell
.\OXN\nsos\build-mvp\Release\nsos_api_server.exe `
  --host 127.0.0.1 `
  --allow-unauthenticated-local
```

The server refuses non-loopback binds without a token.

### Hardening Flags

- `--max-body-bytes`
- `--max-header-bytes`
- `--max-queue-depth`
- `--socket-timeout-ms`
- `--rate-limit-rpm`
- `--max-generate-tokens`
- `--max-batch-prompts`
- `--max-request-context`
- `--health-auth`
- `--trust-proxy-headers`
- `--require-tls-proxy-header`

### Admin Endpoints

`/train-text`, `/train-batch`, `/train-corpus`, and `/pack` are disabled by
default. Start with `--enable-admin-endpoints` to allow them.

`/pack` writes only under `--pack-root` unless
`--allow-pack-absolute-paths` is explicitly set.

```powershell
Invoke-RestMethod http://127.0.0.1:8080/generate `
  -Method Post `
  -ContentType 'application/json' `
  -Headers @{ Authorization = 'Bearer change-me' } `
  -Body '{"prompt":"hello","max_tokens":16,"temperature":0.7}'
```

## Python Binding

```powershell
$env:PYTHONPATH = ".\OXN\nsos\build-mvp\Release"
python -c "import nsos_ext; c=nsos_ext.ModelConfig(); e=nsos_ext.InferenceEngine(); assert e.load_model('', c); print(e.generate('hi', 1, 0.0))"
```

Generated strings are sanitized to valid UTF-8 at the tokenizer/API boundary.

## Docker

```bash
docker build -t nsos-mvp .
docker run --rm -p 8080:8080 -e NSOS_API_TOKEN=change-me nsos-mvp
```

The container binds `0.0.0.0` and therefore requires `NSOS_API_TOKEN`.

## GPU Builds and Diagnostics

`NSOS_GPU_BACKEND` accepts `AUTO`, `NONE`, `CUDA` or `HIP`. The complete AMD
SDK setup, CUDA-preserving build commands, device selection, RX 7600 training
profile and DLL troubleshooting are in
[docs/AMD_GPU_BACKEND.md](docs/AMD_GPU_BACKEND.md).

```powershell
ctest --test-dir .\OXN\nsos\build-codex-hip -j 1 --output-on-failure
```

The runtime reports the compiled backend and visible devices through
`nsos_ext.gpu_backend_name()` and `nsos_ext.gpu_devices()`.
