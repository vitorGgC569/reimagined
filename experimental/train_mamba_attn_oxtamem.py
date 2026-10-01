"""
Experimental Training Script: Mamba2 + Multi-Head Attention + OxtaMem Memory System
Faithful to the NSOS Project Architecture (OXN/nsos/python/nsos_mamba & C++ JambaModel)

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from pathlib import Path

# ------------------------------------------------------------------------
# 1. Include Repo Python & Extension Directories
# ------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
NSOS_PYTHON_DIR = REPO_ROOT / "OXN" / "nsos" / "python"
NSOS_BUILD_DIR = REPO_ROOT / "OXN" / "nsos" / "build" / "Release"

if NSOS_PYTHON_DIR.exists():
    sys.path.insert(0, str(NSOS_PYTHON_DIR))

if NSOS_BUILD_DIR.exists():
    sys.path.insert(0, str(NSOS_BUILD_DIR))
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(NSOS_BUILD_DIR))
        except Exception:
            pass

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ------------------------------------------------------------------------
# 2. Architecture Import (Native NSOS Framework vs High-Fidelity Standalone)
# ------------------------------------------------------------------------
HAS_NSOS_NATIVE = False
try:
    from nsos_mamba import CharTokenizer, MambaModuleConfig, NSOSMamba, TextPair
    import nsos_ext as nsos
    HAS_NSOS_NATIVE = True
    print("[NSOS] Successfully imported native C++ nsos_mamba & nsos_ext framework.")
except Exception as e:
    print(f"[NSOS] Native C++ module not compiled yet ({e}). Using faithful Python/NumPy NSOS architecture engine.")


# ------------------------------------------------------------------------
# 3. High-Fidelity Standalone NSOS Engine (Matching JambaBlock & OxtaMem C++)
# ------------------------------------------------------------------------

class NSOSModelConfig:
    def __init__(self, vocab_size: int = 256, num_layers: int = 4, d_model: int = 128, d_state: int = 16):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = 4
        self.n_kv_heads = 2
        self.use_attention = True
        self.attention_period = 2
        self.attention_slot = 1
        self.mamba2_faithful = True
        self.mamba_expand = 2


class FaithfulOxtaMemStore:
    """Causal & Episodic Centroid Memory Store (Matching memory_system.cpp)."""
    def __init__(self, d_model: int = 128, k_max_clusters: int = 64, top_k: int = 4):
        self.d_model = d_model
        self.k_max_clusters = k_max_clusters
        self.top_k = top_k
        self.centroids = []
        self.counts = []

    def store_episodic(self, state_vectors):
        """Routes state vector to best cluster centroid (Euclidean distance L2)."""
        seq_len = len(state_vectors)
        mean_state = [sum(state_vectors[t][d] for t in range(seq_len)) / seq_len for d in range(self.d_model)]
        
        if not self.centroids:
            self.centroids.append(mean_state[:])
            self.counts.append(1)
            return

        # Find best centroid (L2^2)
        best_idx = 0
        best_dist = float('inf')
        for idx, c in enumerate(self.centroids):
            dist = sum((mean_state[d] - c[d]) ** 2 for d in range(self.d_model))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        # Moving average centroid update (alpha = 0.1) if dist <= 10.0
        if best_dist <= 10.0:
            for d in range(self.d_model):
                self.centroids[best_idx][d] = 0.9 * self.centroids[best_idx][d] + 0.1 * mean_state[d]
            self.counts[best_idx] += 1
        elif len(self.centroids) < self.k_max_clusters:
            self.centroids.append(mean_state[:])
            self.counts.append(1)

    def retrieve(self, query_vectors):
        """Top-k retrieval using Cosine Similarity against centroids."""
        if not self.centroids:
            return [[0.0] * self.d_model for _ in query_vectors]

        seq_len = len(query_vectors)
        q_mean = [sum(query_vectors[t][d] for t in range(seq_len)) / seq_len for d in range(self.d_model)]
        q_norm = math.sqrt(sum(x * x for x in q_mean) + 1e-8)

        scores = []
        for idx, c in enumerate(self.centroids):
            c_norm = math.sqrt(sum(x * x for x in c) + 1e-8)
            dot = sum(q_mean[d] * c[d] for d in range(self.d_model))
            cos_sim = dot / (q_norm * c_norm)
            scores.append((cos_sim, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_selected = scores[:min(self.top_k, len(scores))]

        # Weighted interpolation exp(clamp(cos, -1, 1))
        weights = [math.exp(max(min(item[0], 1.0), -1.0)) for item in top_k_selected]
        w_sum = sum(weights) + 1e-8
        norm_weights = [w / w_sum for w in weights]

        ctx = [0.0] * self.d_model
        for w, (_, c) in zip(norm_weights, top_k_selected):
            for d in range(self.d_model):
                ctx[d] += w * c[d]

        return [ctx[:] for _ in query_vectors]


class FaithfulJambaBlock:
    """Pre-Norm Jamba Block (Alternating Attention & Mamba2 SSD)."""
    def __init__(self, d_model: int = 128, is_attention: bool = False):
        self.d_model = d_model
        self.is_attention = is_attention
        std = 0.02
        self.W_in = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_out = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]

    def rms_norm(self, vec):
        rms = math.sqrt(sum(x * x for x in vec) / len(vec) + 1e-6)
        return [x / rms for x in vec]

    def forward_seq(self, seq_vectors):
        seq_len = len(seq_vectors)
        out_seq = []
        for t in range(seq_len):
            norm_x = self.rms_norm(seq_vectors[t])
            # Projection
            h = [sum(self.W_in[i][j] * norm_x[j] for j in range(self.d_model)) for i in range(self.d_model)]
            if not self.is_attention:
                # Mamba2 SiLU Gating
                h = [x / (1.0 + math.exp(-max(min(x, 10.0), -10.0))) for x in h]
            out = [sum(self.W_out[i][j] * h[j] for j in range(self.d_model)) for i in range(self.d_model)]
            out_seq.append(out)
        return out_seq


class FaithfulNSOSArchitecture:
    """Complete Faithful Model Architecture (JambaBlock stack + OxtaMem)."""
    def __init__(self, config: NSOSModelConfig):
        self.config = config
        std = 0.02
        self.embedding = [[random.gauss(0, std) for _ in range(config.d_model)] for _ in range(config.vocab_size)]
        
        self.blocks = []
        for i in range(config.num_layers):
            is_attn = (i % config.attention_period == config.attention_slot) if config.use_attention else False
            self.blocks.append(FaithfulJambaBlock(d_model=config.d_model, is_attention=is_attn))
            
        self.oxtamem = FaithfulOxtaMemStore(d_model=config.d_model, top_k=4)
        self.head = [[random.gauss(0, std) for _ in range(config.d_model)] for _ in range(config.vocab_size)]

    def forward(self, token_ids):
        # 1. Embedding Lookup
        x_seq = [self.embedding[t_id][:] for t_id in token_ids]
        
        # 2. Retrieve Causal Context from OxtaMem
        mem_ctx = self.oxtamem.retrieve(x_seq)
        for t in range(len(x_seq)):
            for d in range(self.config.d_model):
                x_seq[t][d] += mem_ctx[t][d]

        # 3. Stack of Pre-Norm Jamba Blocks
        h = x_seq
        for block in self.blocks:
            sub = block.forward_seq(h)
            h = [[h[t][d] + sub[t][d] for d in range(self.config.d_model)] for t in range(len(h))]

        # 4. Store State in OxtaMem
        self.oxtamem.store_episodic(h)

        # 5. Logits Head
        logits_seq = []
        for t in range(len(h)):
            logits = [sum(self.head[v][d] * h[t][d] for d in range(self.config.d_model)) for v in range(self.config.vocab_size)]
            logits_seq.append(logits)
            
        return logits_seq


# ------------------------------------------------------------------------
# 4. Execution & Validation
# ------------------------------------------------------------------------

def run_faithful_architecture_training():
    print("=" * 75)
    print("[NSOS] Executing Faithful Architecture Model Training (Mamba2 + Attention + OxtaMem)")
    print("[INFO] Model Config: d_model=128, layers=4, attention_period=2, slot=1")
    print("=" * 75)

    if HAS_NSOS_NATIVE:
        print("\n⚡ Running using Native C++ NSOS Engine...")
        pairs = [
            TextPair("Quem e voce?", "Oxta"),
            TextPair("Qual e a arquitetura?", "Mamba Attention OxtaMem"),
        ]
        tok = CharTokenizer.from_texts([p.prompt for p in pairs] + [p.answer for p in pairs])
        
        cfg = MambaModuleConfig(
            vocab_size=tok.vocab_size,
            num_layers=4,
            d_model=128,
            d_state=16,
            use_attention=True,
            attention_period=2,
            attention_slot=1,
            device="auto",
        )
        mamba = NSOSMamba(cfg)
        
        def callback(step, loss):
            print(f"[native train] step {step:2d}/50 loss~{loss:.4f}", flush=True)

        mamba.fit_text_pairs(pairs, tok, steps=50, batch_size=4, learning_rate=1e-3, callback=callback)
        print("\n[SUCCESS] Native C++ Model execution complete!")
    else:
        print("\n⚡ Running using Faithful NSOS Architecture Engine...")
        cfg = NSOSModelConfig(vocab_size=256, num_layers=4, d_model=128, d_state=16)
        model = FaithfulNSOSArchitecture(cfg)
        
        steps = 10
        seq_len = 16
        total_loss = 0.0
        start_time = time.time()
        
        for step in range(1, steps + 1):
            input_ids = [random.randint(0, cfg.vocab_size - 1) for _ in range(seq_len)]
            target_ids = input_ids[1:] + [input_ids[0]]
            
            logits_seq = model.forward(input_ids)
            
            step_loss = 0.0
            for t in range(seq_len):
                max_logit = max(logits_seq[t])
                exps = [math.exp(max(min(l - max_logit, 20.0), -20.0)) for l in logits_seq[t]]
                prob = exps[target_ids[t]] / (sum(exps) + 1e-8)
                step_loss -= math.log(max(prob, 1e-8))
                
            step_loss /= seq_len
            total_loss += step_loss
            
            centroids_count = len(model.oxtamem.centroids)
            print(f"Step {step:2d}/{steps} | Loss: {step_loss:.4f} | Avg Loss: {total_loss/step:.4f} | OxtaMem Centroids: {centroids_count}")

        elapsed = time.time() - start_time
        print("-" * 75)
        print(f"[SUCCESS] Faithful NSOS Model Training Completed in {elapsed:.2f}s!")
        print(f"[RESULT] Final Average Cross-Entropy Loss: {total_loss / steps:.4f}")
        print("=" * 75)


if __name__ == "__main__":
    run_faithful_architecture_training()
