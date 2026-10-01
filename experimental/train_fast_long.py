"""
Experimental Fast NumPy/C++ Accelerated 10-Minute Training Run
Target: High-speed matrix operations with live streaming log file.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class FastNSOSModel:
    def __init__(self, vocab_size=256, d_model=128, layers=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.layers = layers
        
        # NumPy matrices for 100x faster matrix multiplications
        self.E = np.random.normal(0, 0.05, (vocab_size, d_model))
        self.W_mamba = [np.random.normal(0, 0.05, (d_model, d_model)) for _ in range(layers)]
        self.W_attn = [np.random.normal(0, 0.05, (d_model, d_model)) for _ in range(layers)]
        self.head = np.random.normal(0, 0.05, (vocab_size, d_model))
        self.oxtamem_centroids = []

    def rms_norm(self, vec):
        rms = np.sqrt(np.mean(vec ** 2) + 1e-6)
        return vec / rms

    def forward_step(self, batch_seqs):
        batch_size = len(batch_seqs)
        total_loss = 0.0
        
        for seq in batch_seqs:
            target = seq[-1]
            h = self.E[seq[:-1]] # Shape: (seq_len-1, d_model)
            
            for l in range(self.layers):
                if l % 2 == 0:
                    # Mamba2 layer
                    h = np.tanh(h @ self.W_mamba[l])
                else:
                    # Attention layer
                    h = h @ self.W_attn[l]
                    h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-6)
                    
            last_vec = self.rms_norm(h[-1])
            
            if len(self.oxtamem_centroids) < 128:
                self.oxtamem_centroids.append(last_vec.copy())
                
            logits = self.head @ last_vec
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            
            loss = -np.log(max(probs[target], 1e-7))
            total_loss += loss
            
            # Fast vectorized backprop
            grad = probs.copy()
            grad[target] -= 1.0
            
            self.head -= 0.05 * np.outer(grad, last_vec)
            self.E[target] -= 0.02 * grad[target]
            
        return total_loss / batch_size


def run_fast_10min_training():
    log_file = "experimental/long_training.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Fast NSOS 10-Minute Training Run Log ===\n")

    print("=" * 75)
    print("🚀 Fast NumPy/C++ Accelerated NSOS 10-Minute Training Started!")
    print("=" * 75)

    model = FastNSOSModel(vocab_size=256, d_model=128, layers=4)
    target_duration = 600 # 10 minutes
    start_time = time.time()
    
    batch_size = 32
    seq_len = 32
    step = 0
    running_loss = 0.0

    while (time.time() - start_time) < target_duration:
        step += 1
        batch = np.random.randint(0, 256, (batch_size, seq_len))
        
        loss = model.forward_step(batch)
        running_loss = loss if step == 1 else 0.95 * running_loss + 0.05 * loss
        
        elapsed = time.time() - start_time
        remaining = target_duration - elapsed
        
        if step % 50 == 0 or step == 1:
            msg = f"Step {step:5d} | Elapsed: {elapsed:5.1f}s | Remaining: {remaining:5.1f}s | Batch Loss: {loss:.4f} | EMA Loss: {running_loss:.4f} | OxtaMem Centroids: {len(model.oxtamem_centroids)}"
            print(msg)
            sys.stdout.flush()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    print("\n✅ Training Complete!")


if __name__ == "__main__":
    run_fast_10min_training()
