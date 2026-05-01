# Architecture Risk Reduction

The biggest product risk is concentrated complexity in a few large files:

- `src/jamba.cpp`
- `src/nsos_sdk.cpp`
- `src/http_api_server.cpp`

Do not split them broadly without coverage. The staged plan is:

## Stage 1: Add Audit And Tests

- add a layer audit collector
- test tensor health/stat collection
- test model pack save/reload paths
- test HTTP auth/admin/path behavior

## Stage 2: Extract Leaf Helpers

Safe extraction candidates:

- model pack manifest helpers from `nsos_sdk.cpp`
- HTTP JSON/path/auth helpers from `http_api_server.cpp`
- edge linear pack helpers from `jamba.cpp`

Each extraction must preserve behavior and keep the current tests passing.

## Stage 3: Split Runtime Components

Only after Stage 1 and 2:

- split Jamba block schedules and block implementations
- split model pack IO from inference orchestration
- split HTTP routing from request parsing and auth

The product gate should stay green after every small extraction.
