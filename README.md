# NSOS Workspace

Monorepo for the NSOS (Neural-Symbolic Operating System) project. Multiple maturity tiers in one tree.

## Validated GPU benchmark â€” July 2026

The reproducible Tesla T4 validation package is versioned in this branch:

- [benchmark index and reproduction guide](docs/benchmarks/README.md);
- [full technical report](docs/benchmarks/jamba_vs_nsos_oxtamem_t4_2026-07-22.md);
- [executed Colab notebook](colab/bench_expanded_validation_t4_2026-07-22.ipynb);
- [raw machine-readable results](docs/benchmarks/results/expanded_validation_t4_2026-07-22.json).

It covers NSOS against the reference Jamba dense and Jamba MoE implementations
on multi-seed MQAR, public bAbI QA1, OxtaMem-assisted QA and sequence-length
extrapolation. Read the report before interpreting performance claims.

Clone the exact validated branch:

```bash
git clone --branch nsos-gpu-phases12 --single-branch https://github.com/vitorGgC569/reimagined.git
cd reimagined
```

## Where to look

| Audience | Start here |
|---|---|
| Reproducing the July 2026 GPU benchmark | [`docs/benchmarks/README.md`](docs/benchmarks/README.md) |
| New contributor | [`OXN/nsos/docs/PRODUCT.md`](OXN/nsos/docs/PRODUCT.md) â€” what is supported, what is not |
| Building / running the MVP | [`OXN/nsos/README.md`](OXN/nsos/README.md) and [`OXN/nsos/docs/RELEASE.md`](OXN/nsos/docs/RELEASE.md) |
| Deploying | [`OXN/nsos/docs/DEPLOY.md`](OXN/nsos/docs/DEPLOY.md) |
| Architecture and risk | [`OXN/nsos/docs/ARCHITECTURE_RISK.md`](OXN/nsos/docs/ARCHITECTURE_RISK.md) |
| Validation status | [`OXN/nsos/docs/NSOS_VALIDATION_STATUS.md`](OXN/nsos/docs/NSOS_VALIDATION_STATUS.md) |
| Research / incubation trees (CHRASS, CART, OXB, KernelOpen, pantheon, â€¦) | [`OXN/nsos/docs/INCUBATION.md`](OXN/nsos/docs/INCUBATION.md) |
| Boundary policy (machine-readable) | [`PROJECT_BOUNDARY.json`](PROJECT_BOUNDARY.json), [`docs/PROJECT_BOUNDARY.md`](docs/PROJECT_BOUNDARY.md) |
| Historical audits, removed scripts | [`legacy/README.md`](legacy/README.md) |

## Tiers (one-liner each)

- **Supported product** â€” `OXN/nsos/`. CPU runtime, HTTP API, CLI, Python binding, model packs.
- **Integrated companion** â€” `modules/oxtamem/`. Loopback-first Rust + Python memory engine.
- **Incubation / research** â€” everything else (`KernelOpen/`, `CHRASS/`, `CART/`, `OXB/`, root `pantheon/`, root `bindings/`, `hardware/`, ad-hoc data folders). See `INCUBATION.md`.
- **Legacy** â€” `legacy/`. Frozen historical artifacts; not built, not tested, not supported.

## License

See [`LICENSE`](LICENSE).

