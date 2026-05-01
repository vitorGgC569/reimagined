# Project Scope Status

## Official Core Today

The part of the repository that is currently treated as the product-facing core is:

- `OXN/nsos`

This is where the validated engineering effort is concentrated:

- model/runtime
- training
- tokenizer
- edge-pack export
- API / CLI / Python bindings
- memory integration
- benchmark harness

## Integrated Companion Module

- `modules/oxtamem`

This module is part of the current NSOS story because:

- it is copied into the monorepo intentionally
- it is built from NSOS
- it is used through FFI and as architectural reference

## Important But Not Yet At The Same Validation Level

- `KernelOpen`
- `Pantheon`
- `CHRASS`
- `CART`
- `OXB`
- `hardware`

These areas still matter, but they are not the main validation target for the current release effort.

## Practical Rule For The Repo

When a task concerns:

- model quality
- edge runtime
- training
- serving
- packs
- memory

the default workspace is `OXN/nsos`.

The rest of the monorepo should be treated as:

- research branches
- supporting infrastructure
- future integration targets
- historical experiments

## What The Repo Still Needs

To feel mature as a monorepo, the repository still needs:

- a clearer root README that points first to NSOS
- a distinction between official and experimental folders
- archival or relocation of legacy reports/scripts that are no longer part of the active path
- one release-oriented top-level document that says what is shipping now
