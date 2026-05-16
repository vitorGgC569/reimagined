# OXN

`OXN/nsos` is the only supported MVP runtime in this workspace.

## Maturity Matrix

| Area | Status | Notes |
| --- | --- | --- |
| `OXN/nsos` CPU runtime | MVP | Build, CLI, HTTP API, Python binding and native tests are the release path. |
| CUDA/GPU | Experimental | Kept behind explicit CUDA builds. `test_gpu_parity` is diagnostic, not a release gate. |
| OxtaMem library/FFI | Beta | Useful as local companion code. Network service is loopback-first and must be authenticated outside loopback. |
| Distributed/MPI | Experimental | Do not present as production multi-node support. |
| TTT flows | Research | Available only where explicitly wired and tested. |
| Lean/self-healing claims | Research | Current code is not production formal verification. |
| Old servers/scripts | Removed | Root `serve_oxn.py` and the legacy root `train_*.py` mocks were removed in the MVP cleanup. Hybrid prototypes (`train_oxn_real.py`, `train_oxn_full.py`, `train_oxn_complete.py`, `train_oxn_wikitext.py`) live under `legacy/scripts/` for historical reference. Use `OXN/nsos` entrypoints. |

## Supported Entry Points

- `OXN/nsos/src/api_server.cpp` -> `nsos_api_server`
- `OXN/nsos/src/nsos_cli.cpp` -> `nsos_cli`
- `OXN/nsos/src/bindings.cpp` -> `nsos_ext`

Read `OXN/nsos/README.md` for build, test and API commands.
