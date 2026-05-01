# Project Scope Status

This file is the human summary. The machine-readable source of truth is
`PROJECT_BOUNDARY.json`.

## Supported Product

- `OXN/nsos`

This is the product-facing NSOS surface. Release claims and product gates should
default here.

Current product scope:

- model/runtime
- training paths that pass the NSOS gate
- tokenizer
- model pack load/save
- API / CLI / Python bindings
- memory integration points
- benchmark and fuzz smoke scripts under `OXN/nsos/scripts`

## Integrated Companion Module

- `modules/oxtamem`

OxtaMem is integrated with NSOS, but remains a companion module with its own
Rust/Python validation lane. It can support the product story without expanding
the supported NSOS API by default.

## Research / Incubation

Everything outside the supported product, integrated module, and release-support
files is incubation by default.

Known incubation areas:

- `KernelOpen`
- root-level Pantheon code under `include`, `src`, and `bindings`
- `CHRASS`
- `CART`
- `OXB`
- `hardware`
- root-level benchmark, training, demo, and legacy test tracks
- datasets and generated experiment inputs

Incubation code can be developed normally. It becomes product or integrated
surface only after passing the promotion gate in `docs/PROJECT_BOUNDARY.md`.

## Practical Rule

When a task concerns model quality, edge runtime, serving, packs, or the current
memory integration, start in `OXN/nsos` and `modules/oxtamem`.

When a task concerns a research folder, keep its claims scoped to incubation
until it has deterministic builds, tests, CI gates, documentation, and an updated
entry in `PROJECT_BOUNDARY.json`.
