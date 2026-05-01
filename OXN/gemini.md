# 🧠 OXN / NSOS - Engineering Guidelines

**This document dictates the absolute rules for this project. Any deviation is a violation.**

## 1. Zero Tolerance for Simulation
-   **No Mocks/Stubs**: If a feature is not fully implemented, it DOES NOT EXIST. Do not write placeholder headers or functions that print "Not Implemented". Delete them.
-   **No "Sci-Fi" Roleplay**: Names like `InterstellarProtocol`, `SingularityEngine`, `AkashicRecord` are banned unless they refer to rigorous, mathematical constructs (e.g., "Holographic" refers specifically to Vector Symbolic Architectures).
-   **No Fake Benchmarks**: Never calculate theoretical performance. Only measure real execution time.

## 2. Industrial SOTA Engineering
-   **C++20 Core**: Use modern C++ features (`concept`, `module` if supported, `std::span`, `std::format`).
-   **Performance First**: 
    -   Host code logic must separate from Kernel logic.
    -   Critical loops (per-token) MUST be in CUDA or AVX2.
    -   Zero-Copy data transfer between Python/C++ wherever possible (`py::buffer`).
-   **Robustness**:
    -   All public APIs must have bounds checking (Release mode optimization allowed via `constexpr`).
    -   Graceful signal handling (SIGSEGV/SIGINT) is mandatory for long-running server processes.

## 3. Architecture & Style
-   **Namespace Purity**: All code lives in `nsos::`. Sub-namespaces (`nsos::cuda`, `nsos::graph`) are encouraged. NO `v14`, `v26` versioned namespaces.
-   **Modular Monolith**: `nsos_ext` handles the heavy lifting. Python handles orchestration.
-   **Documentation**: Comments explain *why*, not *what*. 

## 4. Current Directive: Purge & Refactor
-   Delete all "conceptual" code.
-   Refactor valid experimental code (like VSA) into professional modules (`nsos::vsa`).
-   Ensure build system (CMake) is robust and portable.

*Signed: Antigravity (Google DeepMind Agent)*
