# `legacy/src/`

C++ source files superseded by `_v2` rewrites or refactored away during the MVP cleanup.

## Files

- `fabric.cpp` — original Fabric implementation. Superseded by `OXN/nsos/src/fabric_v2.cpp`. The CMake build references only `fabric_v2.cpp` (see `OXN/nsos/CMakeLists.txt:58`); this file was a confusing leftover in the source tree. Moved here in Phase 0, PR-0.4.

The header `OXN/nsos/include/fabric.h` is still in use (included by `fabric_v2.cpp`), so it stays in the active tree.
