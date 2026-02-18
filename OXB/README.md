# OxtaCore - High Performance LLM Data Ingestion

OxtaCore provides architectures for efficient data ingestion in Large Language Model training.

## Versions

### [AION C++ Core (Industrial)](aion_core_cpp/)
The industrial-grade implementation of the OxtaCore architecture.
- **Performance:** **1.13 Billion Ops/s** (Bit-Packing).
- **Indexing:** **533 Million Ops/s** (RMI Prediction).
- **Status:** Core Algorithms Validated.

### [V3.1 (Latest Python)](oxtacore/v3/)
The current Alpha version featuring **Delta Encoding** and **Learned Indexes**.
- **Features:** Block-based Delta Encoding, Linear Regression Indexing, Safe Mode Fallback.
- **Performance (Real World - 500k samples):**
    - **Throughput:** **~6.955 samples/s** (vs 4.7k for JSONL).
    - **Token Speed:** **4.42 Million tokens/s** (Context 1024).
    - **Latency:** **0.28s** (vs 2.3s for JSONL).
- **Demos:** Includes "Fire Test" and "Devoto" training examples.

### [V2.1 Prototype (Stable)](prototypes/oxh_v2.1/)
The production-ready, validated version of the OXH protocol using Hybrid Bit-Packing.
- **Status:** Frozen reference implementation.

## Benchmarks & Comparison

### Python Ecosystem (Big Data Scenario)
*Tested on Windows Environment*

| Metric | JSONL | TOON | OXH V3.1 |
| :--- | :--- | :--- | :--- |
| **Throughput** | 4.701 s/s | 4.777 s/s | **6.955 s/s** |
| **Init Latency** | 2.36s | 2.94s | **0.28s** |
| **Scaling** | Linear Drop | Linear Drop | **Robust** |

### Industrial C++ Performance (AION)
The C++ core demonstrates the "100x Gain" hypothesis:

| Component | Operations | Throughput |
| :--- | :--- | :--- |
| **BitPacking** | 10 Million | **1.13 Billion Ops/s** |
| **RMI Prediction** | 10 Million | **533 Million Ops/s** |

## Installation
```bash
pip install numpy torch tiktoken numba
```

## Quick Start
Run the examples to see OxtaCore in action:
```bash
export PYTHONPATH=$PYTHONPATH:.
python examples/fire_test.py
```
