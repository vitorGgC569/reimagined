# OxtaMem Integration

`modules/oxtamem` is the integrated companion module for NSOS. It is not the
core product surface, but it participates in the release story through explicit
gates.

## Supported Integration Contract

The current integration contract is:

- Rust engine builds and tests through Cargo
- clippy runs with `-D warnings`
- Python SDK syntax smoke passes
- NSOS C++ integration uses `OxtaMemFFI` when the native library is available
- Product builds may disable OxtaMem with `NSOS_BUILD_OXTAMEM=OFF`

## Product Lane Boundary

The CPU product lane does not require the OxtaMem Rust engine to be built inside
the NSOS CMake build. When the Rust target is absent, C++ FFI tests are not part
of product CTest.

## Integrated Lane Gate

```powershell
cargo test --manifest-path .\modules\oxtamem\oxta_engine\Cargo.toml --all-targets
cargo clippy --manifest-path .\modules\oxtamem\oxta_engine\Cargo.toml --all-targets -- -D warnings
python -m compileall -q .\modules\oxtamem\python .\modules\oxtamem\oxta_engine\python .\modules\oxtamem\nn
```

## Promotion Rule

Any expanded OxtaMem API or service guarantee must add:

- auth/resource-limit tests
- persistence corruption tests
- bounded recall/search tests
- NSOS-side integration tests
- documentation in this file and `PROJECT_BOUNDARY.json` when the boundary changes
