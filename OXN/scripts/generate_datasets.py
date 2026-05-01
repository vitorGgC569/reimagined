import numpy as np
import os
import sys

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

VOCAB_SIZE = 128
SEQ_LEN = 32

def generate_sanity(n_samples=1000):
    """
    Phase 0: Sanity & Calibration
    Task: Next Token Prediction.
    Pattern: [0, 1, 2, 0, 1, 2...]
    Target: Shifted by 1.
    """
    print(f"[Phase 0] Generating Sanity Dataset ({n_samples} samples)...")

    # 1. Cyclic Pattern
    data = np.zeros((n_samples, SEQ_LEN), dtype=np.int32)
    for i in range(n_samples):
        start = np.random.randint(0, 3)
        pattern = np.arange(start, start + SEQ_LEN) % 3
        data[i] = pattern
    np.save("data/sanity.npy", data)

def generate_algo(n_samples=1000):
    """
    Phase 1: Algorithmic Reasoning
    Task: Sorted List.
    Input: [5, 1, 3] -> Target [1, 3, 5]
    """
    print(f"[Phase 1] Generating Algo Dataset ({n_samples} samples)...")
    data = np.zeros((n_samples, SEQ_LEN), dtype=np.int32)
    half = SEQ_LEN // 2
    for i in range(n_samples):
        # Problem: Random
        prob = np.random.randint(0, 10, size=half)
        # Solution: Sorted
        sol = np.sort(prob)
        data[i, :half] = prob
        data[i, half:] = sol
    np.save("data/algo.npy", data)

def generate_long_mem(n_samples=500):
    """
    Phase 2: Long Memory
    Task: Needle Retrieval.
    """
    print(f"[Phase 2] Generating Memory Dataset ({n_samples} samples)...")
    data = np.zeros((n_samples, SEQ_LEN), dtype=np.int32)
    for i in range(n_samples):
        needle = np.random.randint(20, 100)
        data[i, :] = np.random.randint(0, 10, size=SEQ_LEN) # Noise
        data[i, 0] = needle # Key at start
        data[i, -1] = needle # Query at end (Target should be needle)
    np.save("data/memory.npy", data)

def generate_code(n_samples=500):
    """
    Phase 3: Code
    Task: Variable Assignment. A=5; Print A -> 5
    """
    print(f"[Phase 3] Generating Code Dataset ({n_samples} samples)...")
    data = np.zeros((n_samples, SEQ_LEN), dtype=np.int32)
    for i in range(n_samples):
        val = np.random.randint(0, 10)
        # 10=SET, 11=A, val, 12=PRINT, 11=A, 13=EOS
        seq = [10, 11, val, 12, 11, val]
        # Pad rest
        full = np.zeros(SEQ_LEN, dtype=np.int32)
        full[:len(seq)] = seq
        data[i] = full
    np.save("data/code.npy", data)

if __name__ == "__main__":
    generate_sanity()
    generate_algo()
    generate_long_mem()
    generate_code()
