import os
import sys
import torch
import psutil
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def test_stress_memory():
    print("🔬 Running Memory Stress Test (1000 Iterations)...")

    process = psutil.Process()
    rss_start = process.memory_info().rss / 1024 / 1024
    print(f"   Start Memory: {rss_start:.2f} MB")

    # Model Setup
    model = nsos_ext.JambaModel(2, 64, 100) # Small model
    ctx = nsos_ext.Context()
    input_ids = [1, 2, 3, 4] * 16 # 64 tokens

    # Warmup
    for _ in range(10):
        _ = model.forward_ids(input_ids, ctx)

    rss_warmup = process.memory_info().rss / 1024 / 1024
    print(f"   Warmup Memory: {rss_warmup:.2f} MB")

    # Stress Loop
    STEPS = 1000
    start_time = time.time()

    for i in range(STEPS):
        # Forward
        # Note: We must clear ctx or use new one to avoid infinite growth?
        # Context stores tensors. If we reuse 'ctx' pointer in C++, does it clear?
        # JambaBlock::forward keys: "attn_input", "moe_indices", etc.
        # They overwrite. So memory usage should be constant per step.
        # BUT: 'Mamba2SSD::ssd_backward' recomputes history.
        # Let's check if we leak tensors.

        hidden = model.forward_ids(input_ids, ctx)

        # Simulated Backward
        # Create gradient tensor
        grad = nsos_ext.Tensor([64, 64], nsos_ext.Device.CPU) # [N, D]
        model.backward_external(grad, ctx)

        if i % 100 == 0:
            current_rss = process.memory_info().rss / 1024 / 1024
            # print(f"   Step {i}: {current_rss:.2f} MB")

    end_time = time.time()
    rss_end = process.memory_info().rss / 1024 / 1024

    print(f"   End Memory: {rss_end:.2f} MB")
    print(f"   Growth: {rss_end - rss_warmup:.2f} MB")
    print(f"   Throughput: {STEPS / (end_time - start_time):.2f} steps/sec")

    # Threshold: Allow small python overhead (fragmentation), e.g. 50MB
    if (rss_end - rss_warmup) > 50.0:
        print("❌ MEMORY LEAK DETECTED (> 50MB growth)")
        sys.exit(1)
    else:
        print("✅ Memory Stable.")
        sys.exit(0)

if __name__ == "__main__":
    test_stress_memory()
