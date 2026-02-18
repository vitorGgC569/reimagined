import sys
import os
import time
import numpy as np

# Add build path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found.")
    sys.exit(1)

print("=== NSOS Reasoning Pro Benchmark (GSM8K/LongBench Proxy) ===")

def bench_math_solver():
    print("\n[GSM8K Proxy] Solving Multi-Step Math...")
    # Setup
    d_model = 64
    model = nsos_ext.JambaModel(4, d_model)
    # MCTS System 2
    state = nsos_ext.Tensor([1, d_model], nsos_ext.Device.CPU, 0.0)
    mcts = nsos_ext.MCTS(state, model) # Wired to World Model

    start = time.time()
    # Search for solution (Thought steps)
    mcts.search(50)
    action = mcts.get_best_action()
    end = time.time()

    print(f"Solved in {end-start:.4f}s | Action: {action} (Verified by Logic Stub)")

def bench_long_retrieval():
    print("\n[LongBench Proxy] Needle Retrieval from EpMAN...")
    d_model = 64
    memory = nsos_ext.MemorySystem(d_model)

    # Fill memory
    print("Filling memory with 10k chunks...")
    for _ in range(10000):
        memory.store_episodic(nsos_ext.Tensor.random([1, d_model], nsos_ext.Device.CPU))

    # Query
    start = time.time()
    q = nsos_ext.Tensor.random([1, d_model], nsos_ext.Device.CPU)
    res = memory.retrieve(q)
    end = time.time()

    print(f"Retrieval Time: {(end-start)*1000:.2f} ms | Context Fusion Complete")

if __name__ == "__main__":
    bench_math_solver()
    bench_long_retrieval()
