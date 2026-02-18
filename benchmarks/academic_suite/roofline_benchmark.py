
import time
import numpy as np
import os
import sys
import psutil

# Path Setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found.")
    sys.exit(1)

def run_roofline_analysis():
    print("=== 🚀 HPC Roofline & Performance Analysis ===")

    # 1. Configuration
    B, S, D = 1, 1024, 512
    vocab = 2000
    model = nsos_ext.JambaModel(4, D, vocab) # 4 layers

    # 2. Arithmetic Intensity Estimate
    # Mamba2 FLOPs per step approx:
    # Projections: 3 * D * (2D) * 2 (MACs)
    # Scan: Linear in D, negligible compared to Matmul
    # Total ~ 12 * D^2 per token
    flops_per_token = 12 * (D**2)

    # Memory Traffic (IO)
    # Weights: ~ 12 * D^2 parameters (assuming standard linear)
    # Activations (Read Input, Write Output): 2 * D * sizeof(float)
    # In inference (decoding), we read weights EVERY step.
    bytes_per_token = (12 * (D**2)) * 1.58 / 8 # 1.58-bit weights!
    # Wait, BitNet weights are packed.
    # Size = 12 * D^2 * 2 bits / 8 = 3 * D^2 bytes.
    # Input/Output = 2 * D * 4 bytes (float32)

    # Arithmetic Intensity = FLOPs / Bytes
    # AI = (12 * D^2) / (3 * D^2) = 4 FLOPs/Byte
    # This puts us in "Compute Bound" territory for CPU AVX2 (usually knee is ~0.5-1.0)
    # but "Memory Bound" for GPU Tensor Cores.

    print(f"Model Config: L=4, D={D}, S={S}")
    print(f"Theoretical AI: {flops_per_token / (3*D**2 + 8*D):.2f} FLOPs/Byte")

    # 3. Measurement Loop (TTFT & Throughput)
    input_ids = [np.random.randint(0, vocab) for _ in range(S)]

    # Warmup
    ctx = nsos_ext.Context()
    emb = model.embedding.forward(input_ids[:10]).reshape([1, 10, D])
    model.forward(emb, ctx)

    # A. Time-To-First-Token (Prefill)
    start = time.time()
    emb = model.embedding.forward(input_ids).reshape([1, S, D])
    _ = model.forward(emb, ctx)
    ttft = (time.time() - start) * 1000

    print(f"TTFT (Prefill {S} tokens): {ttft:.2f} ms")

    # B. Generation Throughput (Decoding)
    # Measure 100 steps
    gen_steps = 100
    start = time.time()

    # Use D2FDecoder or manual loop
    # We simulate step-by-step forward
    dummy_input = model.embedding.forward([1]).reshape([1, 1, D])

    for _ in range(gen_steps):
        model.forward(dummy_input, ctx)

    duration = time.time() - start
    tps = gen_steps / duration

    print(f"Generation Throughput: {tps:.2f} tokens/sec")

    # C. Memory Bandwidth Utilization
    # Bytes moved per second = (Weights Size) * TPS
    # Weights Size (MB)
    weights_mb = (3 * D**2 * 4) / (1024*1024) # 4 layers * 3*D^2 bytes
    bw_usage = weights_mb * tps

    print(f"Est. Memory Bandwidth Usage: {bw_usage:.2f} MB/s")

    # D. Peak Memory State
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Peak RAM Usage: {mem_mb:.2f} MB")

    # Constant State Proof?
    # Run 1000 more steps
    start_mem = process.memory_info().rss
    for _ in range(100):
        model.forward(dummy_input, ctx)
    end_mem = process.memory_info().rss

    print(f"Memory Delta (100 steps): {(end_mem - start_mem)/1024:.2f} KB (Should be ~0 for Mamba)")

if __name__ == "__main__":
    run_roofline_analysis()
