# `legacy/scripts/`

Python scripts archived from the repo root during the MVP cleanup (Phase 0, PR-0.1).

## Why these files are here

- **Hybrid prototypes** — `train_oxn_real.py`, `train_oxn_full.py`, `train_oxn_complete.py`, `train_oxn_wikitext.py` use the real `nsos_ext` C++ kernel (`JambaModel`, `forward_ids`, `embedding.backward`, `parameters()`) but bypass the canonical `Trainer.train_step()` Adam path exposed in `bindings.cpp`. They are useful as historical references for how the binding was used before `OXN/nsos/scripts/train_curriculum.py` became the supported training entry point.

## What is **not** here (deleted, not archived)

These files were removed in PR-0.1 because they were pure mocks (synthetic loss, no real C++ kernel call, or unauthenticated demo server):

- `train_oxn_v5.py`, `train_oxn_v6.py`, `train_oxn_v7.py`
- `train_neural_network.py`
- `train_chip_router.py`
- `train_sovereign.py`
- `train_fixed.py`
- `serve_oxn.py`

Recover from `git log` if needed.

## Supported alternatives

| Use case                        | Use this instead                                         |
|---------------------------------|----------------------------------------------------------|
| Train a small NSOS model        | `OXN/nsos/scripts/train_curriculum.py`                   |
| Industrial training entry       | `OXN/nsos/trainer_industrial.py`                         |
| Serve a model via HTTP API      | `nsos_api_server` (built from `OXN/nsos/src/api_server.cpp`) |
| CLI inference                   | `nsos_cli` (built from `OXN/nsos/src/nsos_cli.cpp`)      |
| Python integration              | `import nsos_ext` (built from `OXN/nsos/src/bindings.cpp`) |
