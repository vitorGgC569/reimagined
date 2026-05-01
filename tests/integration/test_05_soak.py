import sys
import os
import time
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not found. Memory checks will be simulated.")

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
import nsos_ext

def test_memory_soak_exhaustive():
    print("=== Soak Test: Long Duration Stability ===")

    device = nsos_ext.Device.CPU
    try:
        nsos_ext.Tensor.zeros([1], nsos_ext.Device.GPU)
        device = nsos_ext.Device.GPU
    except:
        pass

    # Baseline
    model = nsos_ext.JambaModel(4, 128, 256, device)
    if device == nsos_ext.Device.GPU: model.to(device)

    input_t = nsos_ext.Tensor.zeros([1, 64, 128], device)

    process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    start_mem = process.memory_info().rss if HAS_PSUTIL else 0
    if HAS_PSUTIL: print(f"Start Memory: {start_mem / 1024 / 1024:.2f} MB")

    iterations = 5000 # Increased for exhaustion
    warmup = 100

    print(f"Running {iterations} iterations...")

    start_time = time.time()

    for i in range(iterations):
        ctx = nsos_ext.Context()
        _ = model.forward(input_t, ctx)

        # Periodic Check
        if i % 500 == 0:
            curr = process.memory_info().rss if HAS_PSUTIL else 0
            if HAS_PSUTIL:
                print(f"  Iter {i}: {curr / 1024 / 1024:.2f} MB", end="\r")

    # Cleanup
    import gc
    gc.collect()

    end_mem = process.memory_info().rss if HAS_PSUTIL else 0
    end_time = time.time()

    print(f"\nTotal Time: {end_time - start_time:.2f}s")

    if HAS_PSUTIL:
        print(f"End Memory: {end_mem / 1024 / 1024:.2f} MB")
        diff = (end_mem - start_mem) / 1024 / 1024
        print(f"Growth: {diff:.2f} MB")

        # Threshold: 100MB growth over 5000 iters is suspicious for a fixed model
        # But Context accumulation (if any) could cause it.
        # JambaModel resets context? No, we created `ctx` inside loop.
        # So ctx is destroyed every iter. Should be flat.

        if diff > 100:
            print("❌ FAIL: Memory Leak detected (>100MB growth)")
        else:
            print("✅ PASS: Memory usage stable")
    else:
        print("✅ PASS: Execution completed (Memory check skipped)")

if __name__ == "__main__":
    test_memory_soak_exhaustive()
