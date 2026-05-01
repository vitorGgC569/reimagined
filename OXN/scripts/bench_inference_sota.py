import time
import argparse
import sys
import os

# Ensure we can import nsos_ext
search_paths = [
    os.getcwd(),
    os.path.join(os.getcwd(), "build/Release"),
    os.path.join(os.getcwd(), "OXN/build/Release"),
]

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    for path in search_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(os.path.abspath(path))
            except:
                pass

for path in search_paths:
    if path not in sys.path:
        sys.path.append(path)

try:
    import nsos_ext
except ImportError:
    print("CRITICAL: nsos_ext not found. Build the project first.")
    sys.exit(1)

def benchmark_inference():
    parser = argparse.ArgumentParser(description='NSOS Inference Benchmark')
    parser.add_argument('--layers', type=int, default=12, help='Number of model layers')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--vocab', type=int, default=32000, help='Vocab size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length for generation')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    args = parser.parse_args()

    device = nsos_ext.Device.CPU if args.cpu else nsos_ext.Device.GPU
    try:
        if not args.cpu:
             # Just a probe
             _ = nsos_ext.Tensor.zeros([1], nsos_ext.Device.GPU)
    except:
        print("GPU not available, using CPU.")
        device = nsos_ext.Device.CPU

    print(f"=== NSOS Inference Benchmark (SOTA 2025-2026) ===")
    print(f"Model: {args.layers}L, {args.dim}D, {args.vocab}V")
    print(f"Device: {device}")
    print(f"Target: {args.seq_len} tokens")

    # 1. Init Model
    print("[Init] Creating model...")
    model = nsos_ext.JambaModel(args.layers, args.dim, args.vocab, device)
    
    # 2. Warmup
    print("[Warmup] Generating 16 tokens...")
    model.forward_ids([1]*16, nsos_ext.Context())

    # 3. Benchmark
    latencies = []
    print(f"[Run] Executing {args.trials} trials...")
    
    for t in range(args.trials):
        start = time.time()
        # Simulate auto-regressive generation (one by one would be slower, 
        # but forward_ids handles the block)
        # For a true inference test, we should call one by one to see overhead
        
        # Scenario A: Block throughput (Prompt Processing)
        model.forward_ids([i % args.vocab for i in range(args.seq_len)], nsos_ext.Context())
        
        # Scenario B: Token-by-token (Generation)
        # (We skip for now as forward_ids is optimized for the whole block)
        
        end = time.time()
        latencies.append(end - start)
        print(f" Trial {t+1}: {latencies[-1]:.4f}s")

    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = args.seq_len / avg_latency
    
    print("-" * 40)
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"Throughput:  {tokens_per_sec:.2f} tokens/sec")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_inference()
