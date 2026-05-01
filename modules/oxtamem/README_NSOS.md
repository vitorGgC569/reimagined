# OxtaMem In NSOS

This folder is the internal copy of `Oxta-Mem` used for NSOS integration work.

What changed here:
- The original repository was copied into `modules/oxtamem` so the source reference remains untouched.
- The Cargo binary target was renamed to `oxta_mem_server` to avoid Windows/PDB output collisions during integrated builds.
- NSOS now builds this Rust engine through the `oxtamem_engine` target when Cargo is available.

How it fits NSOS today:
- `modules/oxtamem/oxta_engine` is the preserved Rust engine and SDK surface.
- `OXN/nsos/include/causal_memory_store.h` and `OXN/nsos/src/causal_memory_store.cpp` are the native NSOS causal-store implementation inspired by the same append-only lineage model.
- `OXN/nsos/src/oxtamem_ffi.cpp` loads the Rust library directly and bridges write/read/recall into NSOS.
- `OXN/nsos/src/memory_system.cpp` now persists conversation history and episodic states into the native causal store and can switch on the direct OxtaMem FFI backend.

Current intent:
- Keep the copied Rust module available as a standalone engine and as a directly callable runtime backend.
- Ship a native C++ causal-memory path inside NSOS v1 while using the Rust engine where FFI integration is coherent.
- Do not expose the RESP server as an NSOS MVP public service. It is loopback-first and requires auth when bound outside loopback.
