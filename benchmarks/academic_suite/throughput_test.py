
import time
import torch
import numpy as np
import os
import sys
import psutil

# Add root and build to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found. Build extensions first.")
    sys.exit(1)

def measure_throughput():
    print("=== 🏎️  Throughput Benchmark (Tokens/sec) ===")

    # Configurations to test
    configs = [
        {"B": 1, "S": 128, "D": 256},
        {"B": 4, "S": 256, "D": 256},
        {"B": 8, "S": 512, "D": 256},
    ]

    vocab = 1000

    print(f"{'Batch':<6} | {'SeqLen':<8} | {'Dim':<6} | {'Tokens/s':<10} | {'Latency(ms)':<12}")
    print("-" * 55)

    for cfg in configs:
        B, S, D = cfg["B"], cfg["S"], cfg["D"]
        model = nsos_ext.JambaModel(2, D, vocab)

        # Prepare Dummy Input
        # Note: nsos_ext Tensor expects flat vector or specific shape depending on binding
        # The embedding forward usually takes [B*S] integers.
        total_tokens = B * S
        input_ids = [np.random.randint(0, vocab) for _ in range(total_tokens)]

        # Warmup
        for _ in range(2):
            emb = model.embedding.forward(input_ids).reshape([B, S, D])
            ctx = nsos_ext.Context()
            _ = model.forward(emb, ctx)

        # Measurement Loop
        iterations = 10
        start_time = time.time()

        for _ in range(iterations):
            # We measure Forward Pass (Inference-like)
            emb = model.embedding.forward(input_ids).reshape([B, S, D])
            ctx = nsos_ext.Context()
            _ = model.forward(emb, ctx)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations
        tokens_per_sec = (total_tokens) / avg_time

        print(f"{B:<6} | {S:<8} | {D:<6} | {tokens_per_sec:<10.2f} | {avg_time*1000:<10.2f}")

if __name__ == "__main__":
    measure_throughput()
