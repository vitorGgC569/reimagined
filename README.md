# NSOS Workspace

Monorepo for the NSOS (Neural-Symbolic Operating System) project. Multiple maturity tiers in one tree.

## Where to look

| Audience | Start here |
|---|---|
| New contributor | [`OXN/nsos/docs/PRODUCT.md`](OXN/nsos/docs/PRODUCT.md) — what is supported, what is not |
| Building / running the MVP | [`OXN/nsos/README.md`](OXN/nsos/README.md) and [`OXN/nsos/docs/RELEASE.md`](OXN/nsos/docs/RELEASE.md) |
| Deploying | [`OXN/nsos/docs/DEPLOY.md`](OXN/nsos/docs/DEPLOY.md) |
| Architecture and risk | [`OXN/nsos/docs/ARCHITECTURE_RISK.md`](OXN/nsos/docs/ARCHITECTURE_RISK.md) |
| Validation status | [`OXN/nsos/docs/NSOS_VALIDATION_STATUS.md`](OXN/nsos/docs/NSOS_VALIDATION_STATUS.md) |
| Research / incubation trees (CHRASS, CART, OXB, KernelOpen, pantheon, …) | [`OXN/nsos/docs/INCUBATION.md`](OXN/nsos/docs/INCUBATION.md) |
| Boundary policy (machine-readable) | [`PROJECT_BOUNDARY.json`](PROJECT_BOUNDARY.json), [`docs/PROJECT_BOUNDARY.md`](docs/PROJECT_BOUNDARY.md) |
| Historical audits, removed scripts | [`legacy/README.md`](legacy/README.md) |

## Tiers (one-liner each)

- **Supported product** — `OXN/nsos/`. CPU runtime, HTTP API, CLI, Python binding, model packs.
- **Integrated companion** — `modules/oxtamem/`. Loopback-first Rust + Python memory engine.
- **Incubation / research** — everything else (`KernelOpen/`, `CHRASS/`, `CART/`, `OXB/`, root `pantheon/`, root `bindings/`, `hardware/`, ad-hoc data folders). See `INCUBATION.md`.
- **Legacy** — `legacy/`. Frozen historical artifacts; not built, not tested, not supported.

## License

See [`LICENSE`](LICENSE).
