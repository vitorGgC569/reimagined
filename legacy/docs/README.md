# `legacy/docs/`

Historical documents archived from repo root during the MVP cleanup (Phase 0, PR-0.2).

## Why these are here

Multiple competing audit reports, technical analyses, and roadmaps had accumulated at the repo root. As of the MVP cleanup, the **single source of truth** for product status is `OXN/nsos/docs/PRODUCT.md` (authored in PR-0.3). These archived docs are useful for:

- Historical context on what was discovered, when.
- Tracing why specific decisions were made (e.g., why MoE bypass was added then removed).
- Validating that current code addresses past audit findings.

## What is here

### Audits and remediation reports
- `RELATORIO_AUDITORIA_MVP_2026-04-26.md` — main MVP audit (Portuguese), drove the hardening work.
- `RELATORIO_REMEDIACAO_MVP_2026-04-26.md` — companion remediation report.
- `RELATORIO_AUDITORIA_TECNICA.md` — earlier technical audit.
- `RELATORIO_TECNICO_FINAL.md`, `relatorio_final_v1.md` — final technical reports per cycle.
- `OXN_TECHNICAL_AUDIT_REPORT.md` — Antigravity team audit.
- `analise.md`, `analise_viabilidade_chrass.md` — exploratory analyses.

### Roadmaps (superseded by `OXN/nsos/docs/PRODUCT.md` and the active `plan` file)
- `ROADMAP_V1_TO_V10.md`
- `ROADMAP_V10_TO_V40.md`
- `ROADMAP_V10_TO_V40_ENGINEERING.md`
- `INDUSTRIAL_ROADMAP.md`
- `roadmapReleaseV1.md`

### Vision and theory
- `project.md` — original architectural vision (NSOS-X1 ambitions, target metrics).
- `OxtaV1Release.md` — V1 technical manual (Body / Mind / Spirit framing).

## Rules (inherited from `legacy/README.md`)

- Append-only: do not edit these files.
- Not on any execution path: nothing builds, tests, or links from here.
- Recovery via `git log` if a deleted (not archived) file is needed.
