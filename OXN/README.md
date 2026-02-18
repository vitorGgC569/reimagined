# NSOS: Neural Symbolic Operating System (v2.0)

**A "Metal" Inference & Training Engine for the Edge.**

NSOS is a high-performance, C++ based Deep Learning framework designed to run Large Language Models (LLMs) on constrained hardware with extreme efficiency. v2.0 introduces **System 2 Reasoning** and **Self-Healing** capabilities.

## 🚀 Features

*   **BitNet b1.58:** Extreme quantization (ternary weights) reduces memory usage by 10x.
*   **Jamba Architecture:** Hybrid Mamba-2 (SSM) + Attention backbone for infinite context handling.
*   **Zero Dependencies:** Core engine is pure C++ ("Metal"). No PyTorch/Python required for inference.
*   **Industrial Ready:** Real MPI support for distributed training, AdamW/Muon optimizers, and robust gradient clipping.
*   **Reasoning (System 2):** Integrated Latent MCTS for tree-search based planning and output generation.
*   **Self-Healing:** Automatic symbolic verification (`LeanVerifier`) to detect and correct logic errors.
*   **Python Bindings:** Use `import nsos_ext` to control the C++ engine from Python comfortably.

## 🛠️ Build & Install

### Prerequisites
*   **CMake** (3.10+)
*   **C++ Compiler** (GCC 10+, Clang 12+, or MSVC 2019+)
*   **Python 3.8+** (for bindings)
*   *(Optional)* **CUDA Toolkit** (for GPU acceleration)
*   *(Optional)* **MPI** (for multi-node training)

### Windows
Double-click `scripts/build_windows.bat` or run:
```cmd
cd scripts
build_windows.bat
```

### Linux / MacOS
```bash
mkdir build && cd build
cmake ..
make -j
```

## 🐍 Python SDK Usage

Once built, you can use the high-level `InferenceEngine` directly from Python:

```python
import nsos_ext

# 1. Configure
config = nsos_ext.ModelConfig()
config.d_model = 512
config.num_layers = 12
config.use_quantization = True

# 2. Load Engine
engine = nsos_ext.InferenceEngine()
if engine.load_model("path/to/weights", config):
    print("Model Loaded!")

# 3. Generate with System 2 Reasoning
response = engine.generate("Solve this logic puzzle...", max_tokens=100)
print(response)

# 4. Self-Healing
# Automatically checks logic and trains if invalid
engine.self_heal("1 + 1 =", "3") # Triggers correction

# 5. Train (On-Device!)
loss = engine.train_step("The quick brown fox jumps over the lazy dog")
print(f"Learning... Loss: {loss}")
```

## 📦 Converting Models

You can convert standard HuggingFace models to run on NSOS:

```bash
pip install torch transformers numpy
python3 scripts/convert_hf_to_nsos.py --model meta-llama/Meta-Llama-3-8B --output ./my_nsos_model
```

## 🧠 Architecture Details

*   **Stability:** We use rigorous RMSNorm and Gradient Clipping (Max Norm 1.0) to prevent NaN explosions.
*   **Optimization:** The SDK implements **AdamW**, **Muon**, and **Sophia** optimizers.
*   **Inference:** Supports CPU (AVX2), GPU (CUDA), and Edge (Scalar fallback) execution modes.

## 📄 License
MIT License. **Code is Sovereign.** You own the engine.
