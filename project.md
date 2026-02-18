# NSOS: Neural-Symbolic Operating System (Project OXN)

## 1. Project Overview & NSOS-X1 Roadmap
**NSOS (Neural-Symbolic Operating System)** is a high-performance, hybrid AI engine designed to bridge the gap between "System 1" (fast, intuitive, neural) and "System 2" (slow, logical, symbolic) thinking. 

### 🚀 The "Best Model in the World" Vision: NSOS-X1
NSOS-X1 is the flagship implementation aiming for global SOTA benchmarks:
- **Architecture:** Jamba-Hybrid (Transformer + Mamba + MoE).
- **Scale:** 100B Parameters (25B active via MoE).
- **Context:** 2M+ effective tokens via Holographic Associative Memory (HAM).
- **Efficiency:** 1.58-bit ternary quantization (10x memory efficiency).
- **Adaptability:** Test-Time Training (TTT) Meta-learning for real-time acquisition.
- **Reasoning:** System 2 integration with MCTS (Yggdrasil) and CHRASS (Dynamic Graph Reasoning).

#### 📊 Target Metrics
| Dimension | Metric | Target NSOS-X1 | Current SOTA |
|-----------|--------|----------------|--------------|
| Knowledge | MMLU   | >90%           | ~86% (GPT-4) |
| Math      | GSM8K  | >95%           | ~90% (o1)    |
| Code      | HumanEval | >95%        | ~92% (C3.5)  |
| Context   | Needle | 100% at 2M     | 1M (Gemini)  |

The core philosophy remains: move *all* heavy computation into compiled C++20 to maximize efficiency and bypass the Python Global Interpreter Lock (GIL).

## 2. Technology Stack

### Core Engine
- **Language:** C++20 (Standard conformant)
- **Build System:** CMake (3.18+) with MSVC/GCC/NVCC support.
- **Bindings:** PyBind11 (seamless C++/Python interop).
- **Parallelism:** OpenMP (CPU), CUDA (GPU), MPI (Distributed).
- **Math Backend:** Custom `Tensor` class with SIMD (AVX2/AVX512) and CUDA kernels.

### Neural Architecture (System 1)
- **Jamba-Hybrid Architecture:** A novel combination of Mamba, Transformer, and MoE layers.
- **Mamba2SSD (State Space Duality):** 
    - Implements linear-time sequence modeling.
    - **Robust/Sensitive Split:** Uses 1.58-bit quantization for robust features (`z`, `x`) and FP32 for sensitive dynamics (`dt`, `B`, `C`).
- **TTT (Test-Time Training):**
    - A layer that *learns* during inference.
    - **Hamiltonian Dynamics:** Uses momentum (`velocity`), energy preservation (`friction`), and annealing (`temperature`) to update weights `W_t` on-the-fly based on the input sequence.
- **MoE (Mixture of Experts):**
    - **Router:** BitLinear gate to select top-k experts.
    - **Experts:** `BitFastKAN` (Kolmogorov-Arnold Networks) using Radial Basis Functions (RBF) and quantized base weights.
- **Attention:** 
    - Optimized `BitLinear` (1.58-bit) projections (`q`, `k`, `v`, `o`) to reduce memory footprint.

### Symbolic Reasoning (System 2)
- **MCTS (Yggdrasil Engine):**
    - Native C++ Monte Carlo Tree Search.
    - Explores latent space trajectories to maximize a value function.
    - Uses UCT (Upper Confidence Bound for Trees) for exploration/exploitation balance.
- **Dynamic Graph Reasoning (CHRASS):**
    - Constructs a topological graph of tokens based on cosine similarity of their latent embeddings.
    - Runs **SSSP (Single-Source Shortest Path)** algorithms to find logical connections between concepts.
    - Injects "Logical Distance" back into the embeddings via soft saturation (`tanh`), grounding the neural representations in symbolic topology.

### Memory Systems
- **Holographic Associative Memory (HAM):**
    - Based on Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA).
    - **Operations:**
        - `Bind (*)`: Combines concepts (e.g., Shape * Round).
        - `Bundle (+)`: Superimposes information.
        - `Permute (Π)`: Encodes sequence order.
    - **Infinite Context:** Allows storage and retrieval of concepts using high-dimensional vector orthogonality.

## 3. Implementation Details

### C++ Components (`OXN/nsos/src`)

#### Model Classes
- **`JambaModel` (`jamba.cpp`):** Top-level container. Manages layers, embedding, memory, and specialized execution modes (`run_simd_inference`, `run_reasoning_loop`).
- **`Mamba2SSD` (`mamba2.cpp`):** Core recurrent layer. Implements the SSD forward/backward pass with custom gradient calculations.
- **`TTTLayer` (`ttt_layer.cpp`):** Implements the online learning loop. Features an internal optimizer (SGD-like with Hamiltonian terms) that updates `W_hidden` for every token.
- **`BitFastKANLayer` (`kan.cpp`):** Efficient implementation of KANs using grid-based RBF expansion and quantized weights.
- **`Tensor` (`tensor.cpp`):** The workhorse. Handles memory management, device placement (CPU/GPU), and math operations (`matmul`, `add`, `mul`, `cross_entropy`). Includes `__cuda_array_interface__` for zero-copy exchange with PyTorch.

#### Infrastructure
- **`SmartLoader` (`smart_loader.cpp`):** Multi-threaded data loader using `pread` for high-throughput I/O without blocking the main computation thread.
- **`Inspector` (`inspector.cpp`):** Singleton for deep observability.
    - **Panic Handler:** Catches `SIGSEGV` and prints the stack trace of the last active logical block.
    - **Health Monitor:** Scans for NaNs/Infs.
- **`NeuromorphicCompiler` (`neuromorphic.cpp`):** Experimental module to export TTT layers as Spiking Neural Network (SNN) configurations.
- **`MultiNodeOrchestrator` (`nsos_mpi.cpp`):** Abstraction for MPI operations (AllReduce, Broadcast) for distributed training.

#### Optimization
- **Custom Optimizers (`optimizers.cpp`):**
    - **`Muon`:** Momentum + Orthogonalization.
    - **`Sophia`:** Second-order information using Hessian diagonal approximation.
    - **`FOGZO`:** First-Order Gradient w/ Zero Overhead (Experimental).
- **BitLinear (`bitlinear.cpp`):** Implements "1.58-bit" weights (ternary {-1, 0, 1}) for extreme memory efficiency.

### Python Bindings (`nsos_ext`)
The `bindings.cpp` file exposes the entire C++ API to Python. Key features:
- **Zero-Copy Tensor Interop:** Passes pointers between C++ `Tensor` and NumPy/PyTorch without data copying.
- **Exception firewall:** Translates C++ exceptions (runtime_error, out_of_range) into Python exceptions to prevent interpreter crashes.
- **Signal Handling:** Installs custom handlers for `SIGSEGV` and `SIGINT` to provide meaningful error messages on crash.

### Industrial Hardening
- **Determinism:** Global seed control for bit-exact reproducibility.
- **Safety:** Checkpointing with atomic renames.
- **Gatekeeper:** Validation scripts to prevent running on unstable environments.

## 4. Workflows

### Training
1.  **Phase 0 (Sanity):** Validates gradient flow and basic tensor ops (`train_phase0_sanity.py`).
2.  **Phase 0 (GPU):** Verifies CUDA kernels on specific hardware.
3.  **Industrial Training:** Main loop (`train_industrial.py`) utilizing `SmartLoader`, `Muon` optimizer, and `JambaModel`.

### Deployment
- **Docker:** Containerized build with architecture-specific flags (`CUDA_ARCH`).
- **Inference Engine:** Dedicated C++ engine exposed to Python for low-latency generation.

## 5. Directory Structure
```
OXN/
├── nsos/
│   ├── src/          # C++ Source (.cpp)
│   ├── include/      # C++ Headers (.h)
│   ├── python/       # Setup scripts
│   └── bindings.cpp  # PyBind11 definition
├── scripts/          # Python Control Plane (Training, Data)
├── benchmarks/       # Performance evaluation
└── tests/            # Unit and Integration tests
```
