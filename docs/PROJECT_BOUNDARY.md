# Project Boundary

This repository is intentionally a monorepo, but it is not a single maturity
surface. The release line is explicit:

- Product: `OXN/nsos`
- Integrated module: `modules/oxtamem`
- Research/incubation by default: everything else unless promoted through this
  document and `PROJECT_BOUNDARY.json`

`PROJECT_BOUNDARY.json` is the machine-readable source of truth. This document
explains how to interpret it.

## Supported Product

`OXN/nsos` is the current supported product surface. Product-facing claims,
release gates, CI failures, and hardening work are expected to land here first.

The product surface includes:

- C++ runtime and model code
- CLI, HTTP API, and Python binding
- tokenizer, model packs, and deterministic reload behavior
- CPU-first inference and training paths that pass the release gate
- NSOS-owned benchmarks and fuzz smokes under `OXN/nsos/scripts`

## Integrated Module

`modules/oxtamem` is an integrated companion module. It is allowed to participate
in the product story when it passes its own Rust/Python gates and is consumed
through the NSOS integration points.

The module is not the same as the core product. Breaking it should fail the
integrated-module lane, but it does not automatically expand the supported NSOS
API surface.

## Release Support

Some files outside `OXN/nsos` are release support rather than product code:

- CI configuration
- root Dockerfile
- root README and scope documents
- gatekeeper and determinism scripts
- the industrial Python smoke used by the gate
- this boundary policy

These files exist to define, build, validate, or package the supported product.

## Incubation

Incubation code can be ambitious, experimental, and useful. It is not dead code,
but it is not release-supported.

Incubation means:

- no product claim should depend on it
- no compatibility promise is implied
- performance or research claims must stay scoped to the experiment
- failures there do not block the supported product unless a promotion is in
  progress
- build artifacts, generated checkpoints, logs, and local binaries must stay out
  of Git

Known incubation roots include `KernelOpen`, `CHRASS`, `CART`, `OXB`,
`hardware`, root-level Pantheon files, root-level benchmarks, root-level tests,
and data experiments.

## Promotion Gate

A module leaves incubation only after all of these are true:

1. It has a clear product reason and owner.
2. It has deterministic build instructions from a clean checkout.
3. Its unit and integration tests are wired into CI.
4. Runtime/service paths have authentication, resource limits, and failure-mode
   tests where applicable.
5. Benchmark claims are reproducible from committed scripts or documented input
   artifacts.
6. Documentation is updated in `README.md`, `PROJECT_SCOPE_STATUS.md`, this file,
   and `PROJECT_BOUNDARY.json`.
7. The CI boundary check passes.

Until then, keep the module in incubation and avoid wording that makes it sound
like part of the supported NSOS product.
