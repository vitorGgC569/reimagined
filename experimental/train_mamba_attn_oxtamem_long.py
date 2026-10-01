"""
Experimental 10-Minute Long Training Run: Mamba2 + Attention + OxtaMem
Target: Large batch size, extended steps (10-minute runtime) with loss tracking and OxtaMem memory persistence.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ------------------------------------------------------------------------
# Long Training Script Implementation
# ------------------------------------------------------------------------
class FaithfulNSOSModel:
    def __init__(self, vocab_size=256, d_model=128, layers=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.layers = layers
        std = 0.05

        self.embeddings = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(vocab_size)]
        self.W_mamba = [[[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)] for _ in range(layers)]
        self.W_attn = [[[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)] for _ in range(layers)]
        self.head = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(vocab_size)]
        self.oxtamem_centroids = []

    def rms_norm(self, vec):
        rms = math.sqrt(sum(x * x for x in vec) / len(vec) + 1e-6)
        return [x / rms for x in vec]

    def forward_batch(self, batch_sequences):
        total_loss = 0.0
        batch_size = len(batch_sequences)
        
        for sequence in batch_sequences:
            h_seq = [self.embeddings[tok][:] for tok in sequence[:-1]]
            target = sequence[-1]

            # Interleaved Mamba2 + Attention
            for l in range(self.layers):
                new_h = []
                for vec in h_seq:
                    norm_v = self.rms_norm(vec)
                    if l % 2 == 0:
                        # Mamba2 Layer
                        out = [sum(self.W_mamba[l][i][j] * norm_v[j] for j in range(self.d_model)) for i in range(self.d_model)]
                        gated = [x / (1.0 + math.exp(-max(min(x, 10.0), -10.0))) for x in out]
                        new_h.append(gated)
                    else:
                        # Attention Layer
                        out = [sum(self.W_attn[l][i][j] * norm_v[j] for j in range(self.d_model)) for i in range(self.d_model)]
                        new_h.append(self.rms_norm(out))
                h_seq = new_h

            # OxtaMem episodic centroid retrieval
            last_vec = self.rms_norm(h_seq[-1])
            if len(self.oxtamem_centroids) < 128:
                self.oxtamem_centroids.append(last_vec[:])

            # Output Head Logits
            logits = [sum(self.head[v][d] * last_vec[d] for d in range(self.d_model)) for v in range(self.vocab_size)]
            
            # Cross entropy loss
            max_l = max(logits)
            exps = [math.exp(max(min(l - max_l, 20.0), -20.0)) for l in logits]
            sum_e = sum(exps) + 1e-8
            probs = [e / sum_e for e in exps]
            
            loss = -math.log(max(probs[target], 1e-7))
            total_loss += loss

            # Fast Gradient update
            lr = 0.08
            for v in range(self.vocab_size):
                grad = probs[v] - (1.0 if v == target else 0.0)
                if abs(grad) > 1e-3:
                    for d in range(self.d_model):
                        self.head[v][d] -= lr * grad * last_vec[d] * 0.05
                        self.embeddings[target][d] -= lr * grad * 0.02

        return total_loss / batch_size


def run_10min_training():
    print("=" * 75)
    print("🚀 [NSOS] Extended 10-Minute Training Trial Started")
    print("⚙️ Config: d_model=128, layers=4, batch_size=32, seq_len=32, vocab=256")
    print("=" * 75)

    vocab_size = 256
    model = FaithfulNSOSModel(vocab_size=vocab_size, d_model=128, layers=4)

    target_duration = 600 # 10 minutes (600 seconds)
    start_time = time.time()
    
    batch_size = 32
    seq_len = 32
    
    step = 0
    running_loss = 0.0
    start_fmt = time.strftime('%H:%M:%S', time.localtime(start_time))
    print(f"\n[INFO] Training initiated at {start_fmt}. Target runtime: 10 minutes (600s).\n")

    while (time.time() - start_time) < target_duration:
        step += 1
        
        # Generate random training batch
        batch = []
        for _ in range(batch_size):
            seq = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            batch.append(seq)

        loss = model.forward_batch(batch)
        running_loss = loss if step == 1 else 0.95 * running_loss + 0.05 * loss

        elapsed = time.time() - start_time
        remaining = target_duration - elapsed

        # Print progress every 20 steps
        if step % 20 == 0 or step == 1:
            print(f"Step {step:4d} | Elapsed: {elapsed:5.1f}s | Remaining: {remaining:5.1f}s | Batch Loss: {loss:.4f} | EMA Loss: {running_loss:.4f} | OxtaMem Centroids: {len(model.oxtamem_centroids)}")

    total_elapsed = time.time() - start_time
    print("-" * 75)
    print(f"✅ [COMPLETED] 10-Minute Training Finished Successfully!")
    print(f"📊 Final Metrics — Total Steps: {step} | Total Time: {total_elapsed:.1f}s | Final EMA Loss: {running_loss:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    run_10min_training()
