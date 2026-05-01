# NSOS MVP

NSOS is the supported neural runtime in this repository. The MVP is a
CPU-first, authenticated inference service with CLI and Python bindings.

## Supported In MVP

- `nsos_core` CPU runtime.
- `nsos_api_server` HTTP API.
- `nsos_cli` command-line companion.
- `nsos_ext` Python binding.
- Model pack load/save with bounded parsing, manifest version and checksums.
- Tokenizer pack load/save with bounded parsing.
- Native CTest suite plus Python binding smoke test.

## Experimental Or Out Of Scope

- CUDA/GPU runtime is experimental until `test_gpu_parity` is stable on the
  target hardware.
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
  -DNSOS_ENABLE_CUDA=OFF `
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
  -DNSOS_ENABLE_CUDA=OFF \
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

## GPU Diagnostics

CUDA builds are still useful for investigation:

```powershell
$env:NSOS_GPU_PARITY_CASE = "matmul"
.\OXN\nsos\build_cuda129\Release\test_gpu_parity.exe
```

Available cases: `tensor_add`, `matmul`, `rmsnorm`, `bitlinear`,
`mamba_streaming`, `jamba_batch`.
