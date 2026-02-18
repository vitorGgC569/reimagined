import sys
import os
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import nsos_ext
import torch
import numpy as np
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="NSOS Training Script (Cosmopedia)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream dataset")
    args = parser.parse_args()

    print("NSOS Extension Loaded Successfully!")

    # 1. Initialize Model
    print("Initializing Model (Jamba-JEPA-Coconut)...")
    d_model = 64
    model = nsos_ext.JambaModel(4, d_model)
    tokenizer = nsos_ext.Tokenizer()
    vocab_size = 128
    # We use BitFastKANLayer as the Language Head
    head = nsos_ext.BitFastKANLayer(d_model, vocab_size)

    optimizer = nsos_ext.MuonOptimizer([d_model, vocab_size], args.lr) # Fixed dimensions

    # 2. Data Loading
    print("Loading Cosmopedia dataset...")
    try:
        dataset = load_dataset("HuggingFaceTB/cosmopedia", split="train", streaming=args.stream, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to synthetic data.")
        dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(100)]

    # 3. Persistent Embedding Table (Simulation)
    np.random.seed(42)
    embedding_table = np.random.randn(vocab_size, d_model).astype(np.float32)

    # Pre-allocate tensors
    input_tensor = nsos_ext.Tensor([1, d_model], nsos_ext.Device.CPU, 0.0) # Will resize or assume batch 1
    # Note: Binding Tensor resize isn't exposed, so we re-create or fix size.
    # To be safe, we re-create inside loop or use fixed seq len.

    print(f"Starting Training Loop ({args.steps} steps)...")
    iterator = iter(dataset)
    start_time = time.time()

    for step in range(args.steps):
        try:
            sample = next(iterator)
            text = sample['text'][:512]
        except StopIteration:
            iterator = iter(dataset)
            sample = next(iterator)
            text = sample['text'][:512]

        tokens = tokenizer.encode(text)
        if len(tokens) < 2: continue

        # Limit sequence length to avoid massive allocs in demo
        max_seq = 64
        input_ids = [t % vocab_size for t in tokens[:max_seq]]
        target_ids = [t % vocab_size for t in tokens[1:max_seq+1]]
        seq_len = len(input_ids)

        # Look up embeddings
        input_np = embedding_table[input_ids] # [Seq, D]

        # Create Tensor and Load Data
        input_tensor = nsos_ext.Tensor([seq_len, d_model], nsos_ext.Device.CPU, 0.0)
        np.array(input_tensor, copy=False)[:] = input_np # Zero-copy write

        # Forward Pass
        latent = model.forward_embedding(input_tensor) # [Seq, D]

        # Head Forward
        logits_tensor = head.forward(latent) # [Seq, Vocab]
        logits_view = np.array(logits_tensor, copy=False)

        # Gradient Calculation (Simulated Softmax+CrossEntropy Backward)
        # Stable Softmax
        logits_safe = logits_view - np.max(logits_view, axis=1, keepdims=True)
        exp_logits = np.exp(logits_safe)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = np.clip(probs, 1e-9, 1.0)

        grad_np = probs.copy()
        for i, tid in enumerate(target_ids):
            if tid < vocab_size:
                grad_np[i, tid] -= 1.0

        # Backward Prop to Weights
        latent_view = np.array(latent, copy=False)
        # dW = Latent^T * Grad
        dW_np = np.dot(latent_view.T, grad_np) / seq_len

        # Clip
        grad_np = np.clip(grad_np, -1.0, 1.0)
        dW_np = np.clip(dW_np, -1.0, 1.0)

        # Create Grad Tensor for Muon
        dW_tensor = nsos_ext.Tensor([d_model, vocab_size], nsos_ext.Device.CPU, 0.0)
        np.array(dW_tensor, copy=False)[:] = dW_np

        # Step Muon
        optimizer.step_and_quantize(head.base_weight, dW_tensor, head.base_weight)

        if step % 100 == 0:
            # Check generation on "Hello"
            test_ids = tokenizer.encode("Hello")
            t_in = embedding_table[[test_ids[0] % vocab_size]] # [1, D]

            t_tensor = nsos_ext.Tensor([1, d_model], nsos_ext.Device.CPU, 0.0)
            np.array(t_tensor, copy=False)[:] = t_in

            out_lat = model.forward_embedding(t_tensor)
            out_log = head.forward(out_lat)
            out_id = np.argmax(np.array(out_log, copy=False))

            char_out = tokenizer.decode([out_id])
            print(f"Step {step}: Prediction='{char_out}' (id={out_id}) | Loss ~ {np.mean(np.abs(grad_np)):.4f}")

    print("Training simulation complete.")

if __name__ == "__main__":
    main()
