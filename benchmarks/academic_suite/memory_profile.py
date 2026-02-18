
import time
import os
import sys
import psutil
import threading

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found in build/")
    sys.exit(1)

def memory_profile():
    print("=== 🧠 Memory Profile (RSS) ===")

    process = psutil.Process(os.getpid())

    # Baseline
    mem_base = process.memory_info().rss / 1024 / 1024
    print(f"Baseline Memory: {mem_base:.2f} MB")

    # Allocation
    print("Allocating Model (L=12, D=512)...")
    model = nsos_ext.JambaModel(12, 512, 10000)

    mem_model = process.memory_info().rss / 1024 / 1024
    print(f"Model Memory: {mem_model:.2f} MB (Delta: {mem_model - mem_base:.2f} MB)")

    # Inference Run
    print("Running Inference Batch=16, Seq=512...")
    B, S, D = 16, 512, 512
    vocab = 10000
    input_ids = [1] * (B * S)

    emb = model.embedding.forward(input_ids).reshape([B, S, D])
    ctx = nsos_ext.Context()

    _ = model.forward(emb, ctx)

    mem_inf = process.memory_info().rss / 1024 / 1024
    print(f"Inference Peak: {mem_inf:.2f} MB (Delta vs Model: {mem_inf - mem_model:.2f} MB)")

    # Backward (Check for Leaks / History)
    print("Running Backward...")
    grad = nsos_ext.Tensor([B, S, D], nsos_ext.Device.CPU)
    model.backward_external(grad, ctx)

    mem_back = process.memory_info().rss / 1024 / 1024
    print(f"Post-Backward: {mem_back:.2f} MB")

    print("-" * 40)
    if (mem_inf - mem_model) > 500:
        print("⚠️  High Activation Memory Detected!")
    else:
        print("✅ Memory Usage within acceptable bounds (Mamba2 Checkpointing works).")

if __name__ == "__main__":
    memory_profile()
