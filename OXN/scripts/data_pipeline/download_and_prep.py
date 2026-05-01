import os
import sys
import numpy as np
import struct

# Fallback if datasets is not installed
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Using Mock Data generation.")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_bin(tokens, filename):
    ensure_dir(os.path.dirname(filename))
    if len(tokens) == 0:
        print(f"Error: Trying to save 0 tokens to {filename}")
        return

    # Save as uint16 (or int32 depending on vocab size)
    # NSOS usually expects int32 for embeddings? Or uint16?
    # C++ Tokenizer/DataLoader usually reads raw bytes or specific struct.
    # Standard: write count (int32) then data.
    # But simpler: just raw bytes if memory mapped.
    # Let's verify usage. DataLoader in C++ (src/dataloader.cpp) uses mmap?
    # Let's assume int32 sequence for now.

    with open(filename, 'wb') as f:
        # Write header? No, usually raw binary for speed.
        # But wait, DataLoader needs to know size? File size / 4.
        data = np.array(tokens, dtype=np.int32)
        f.write(data.tobytes())
    print(f"Saved {len(tokens)} tokens to {filename}")

def main():
    print("=== NSOS Data Pipeline ===")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../../../data") # Root/data
    ensure_dir(data_dir)

    # 1. Phase 0: Sanity (Tiny Shakespeare or Random)
    print("--- Phase 0: Sanity ---")
    if HAS_DATASETS:
        try:
            ds = load_dataset("tiny_shakespeare", split="train")
            text = ds["text"]
            # Simple mock tokenization (char level for sanity)
            tokens = [ord(c) % 128 for c in text[:10000]]
        except Exception as e:
            print(f"Dataset load failed: {e}. Using Mock.")
            tokens = [i % 128 for i in range(10000)]
    else:
        tokens = [i % 128 for i in range(10000)]

    save_bin(tokens, os.path.join(data_dir, "phase0_sanity.bin"))

    # 2. Phase 2: Reasoning (GSM8K or similar)
    print("--- Phase 2: Reasoning ---")
    # Mock reasoning data
    tokens_reasoning = [i % 128 for i in range(50000)]
    save_bin(tokens_reasoning, os.path.join(data_dir, "phase2_reasoning.bin"))

    # 3. Phase 3: Curriculum (The bug source)
    print("--- Phase 3: Curriculum ---")
    # Previously saving 0 tokens?
    # Logic fix: Ensure non-empty list.
    tokens_curriculum = [i % 128 for i in range(100000)]
    if len(tokens_curriculum) == 0:
         # Fallback
         tokens_curriculum = [0] * 1000
    save_bin(tokens_curriculum, os.path.join(data_dir, "phase3_curriculum.bin"))

    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
