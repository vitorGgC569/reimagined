import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def run_perplexity_eval(model, ctx, data_path, seq_len=32):
    if not os.path.exists(data_path):
        print(f"⚠️  Data not found: {data_path}")
        return float('nan')

    data = torch.load(data_path)

    print("   Measuring Throughput (Tokens/sec)...")
    start_time = time.time()
    total_tokens = 0
    steps = 50

    for _ in range(steps):
        idx = np.random.randint(0, len(data) - seq_len - 1)
        input_ids = data[idx:idx+seq_len].tolist()
        model.forward_ids(input_ids, ctx)
        total_tokens += seq_len

    end_time = time.time()
    throughput = total_tokens / (end_time - start_time)
    print(f"   Throughput: {throughput:.2f} tok/s")
    return throughput

def run_synthetic_reasoning(model, ctx):
    print("   Running Associative Recall Profiling...")

    input_ids = [10, 20] * 10

    # Measure Latent Norm Change
    model.reset_session()

    # We need to handle potential GPU tensor return
    out1_t = model.forward_ids(input_ids[:2], ctx)
    if out1_t.device == nsos_ext.Device.GPU:
        out1 = out1_t.cpu().numpy()
    else:
        out1 = out1_t.numpy()

    norm1 = np.linalg.norm(out1)

    out2_t = model.forward_ids(input_ids[-2:], ctx)
    if out2_t.device == nsos_ext.Device.GPU:
        out2 = out2_t.cpu().numpy()
    else:
        out2 = out2_t.numpy()

    norm2 = np.linalg.norm(out2)

    print(f"   First Pair Norm: {norm1:.4f}, Last Pair Norm: {norm2:.4f}")
    return {"norm_delta": float(abs(norm1 - norm2))}

def run_system2_profile(model):
    print("   Profiling System 2 (Reasoning Loop)...")
    input_ids = [1, 2, 3, 4]
    ctx = nsos_ext.Context()

    # Warmup
    model.forward_ids(input_ids, ctx)

    start = time.time()

    # Ensure latent dim matches model dim (128)
    use_gpu = (model.parameters()[0].data.device == nsos_ext.Device.GPU)
    dev = nsos_ext.Device.GPU if use_gpu else nsos_ext.Device.CPU

    x = nsos_ext.Tensor([1, 4, 128], dev) # Mock latent
    out = model.run_reasoning_loop(x, 4) # 4 steps
    end = time.time()

    latency = (end - start) * 1000.0 # ms
    print(f"   Reasoning Loop (4 steps) Latency: {latency:.2f} ms")
    return latency

def main():
    print("=== Industrial Evaluation Suite ===")

    use_cuda = torch.cuda.is_available()
    nsos_device = nsos_ext.Device.GPU if use_cuda else nsos_ext.Device.CPU
    print(f"   Device: {nsos_device}")

    # Setup
    DIM = 128
    LAYERS = 4
    model = nsos_ext.JambaModel(LAYERS, DIM, 256, device=nsos_device)
    ctx = nsos_ext.Context()

    results = {}

    # 1. Throughput
    data_path = os.path.join(ROOT_DIR, "data/phase_0/train.pt")
    tp = run_perplexity_eval(model, ctx, data_path)
    results["throughput_tok_s"] = tp

    # 2. Reasoning / Adaptation
    adapt = run_synthetic_reasoning(model, ctx)
    results["adaptation_metric"] = adapt

    # 3. System 2 Latency
    lat = run_system2_profile(model)
    results["sys2_latency_ms"] = lat

    # 4. Memory Usage (Mock)
    mem = nsos_ext.InferenceEngine().get_memory_usage()
    results["memory_mb"] = mem

    # Save
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Benchmark Complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    main()
