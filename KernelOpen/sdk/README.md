# Universal Heterogeneous Kernel (UHK) - Python SDK

This directory contains the Python bindings for the UHK C++ runtime, enabling seamless integration of the "Metal" architecture into Python/PyTorch workflows.

## Prerequisites
*   Python 3.7+
*   C++17 Compiler (GCC/Clang/MSVC)
*   `pybind11` (installed via pip)

## Building the SDK

The SDK uses `setuptools` and `pybind11`. To build and install in your current environment:

```bash
# From the project root
pip install .
```

Or for development (editable mode):

```bash
pip install -e .
```

## Usage Example

```python
import numpy as np
import uhk

# Initialize the Persistent Kernel Runtime
runtime = uhk.UniversalKernelRuntime()

# Prepare Data (NumPy Arrays are Zero-Copied if possible)
M, N, K = 128, 128, 128
weights_packed = np.random.randint(0, 255, size=(M * K // 4), dtype=np.uint8)
activations = np.random.randint(-127, 127, size=(K * N), dtype=np.int8)
output = np.zeros(M * N, dtype=np.float32)

# Submit Task
runtime.submit_bitnet_gemm(M, N, K, weights_packed, activations, output)

# Shutdown
runtime.shutdown()
```

## Structure
*   `bindings.cpp`: The bridge between C++ structs and Python objects using `pybind11`.
*   `uhk/`: The Python package directory.
