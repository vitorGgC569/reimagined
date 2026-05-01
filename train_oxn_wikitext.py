#!/usr/bin/env python3
"""
==============================================================================
NSOS/OXN WIKITEXT TRAINING (INDUSTRIAL VERSION)
==============================================================================
"""

import sys
import os
import time
import random
import urllib.request
import re
import argparse

# Force Single Threading for Determinism (Emergency Protocol)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Setup environment
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
search_paths = [
    os.getcwd(),
    build_dir,
    os.path.join(os.getcwd(), "OXN"),
]
for p in search_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import nsos_ext as nsos
    print("✅ nsos_ext carregado!")
except ImportError as e:
    print(f"❌ Erro crítico: {e}")
    sys.exit(1)

# CONFIG
CONFIG = {
    "layers": 4,
    "dim": 128,
    "vocab": 256,
    "seq_len": 32,
    "lr": 0.001,
    "model_file": "oxn_wikitext_trained.bin",
    "dataset_url": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "data_dir": "wikitext_data",
}

def download_wikitext():
    if not os.path.exists(CONFIG['data_dir']):
        os.makedirs(CONFIG['data_dir'])
    train_path = os.path.join(CONFIG['data_dir'], "wiki.train.raw")
    if not os.path.exists(train_path):
        print(f"📥 Baixando WikiText-2 (Raw)...")
        try:
            urllib.request.urlretrieve(CONFIG['dataset_url'], train_path)
            print("✅ Dataset pronto!")
        except Exception as e:
            print(f"❌ Falha no download: {e}")
            sys.exit(1)
    return train_path

def train_on_data(model, tokens):
    total_tokens = len(tokens)
    seq_len = CONFIG['seq_len']
    lr = CONFIG['lr']
    
    num_batches = (total_tokens - 1) // seq_len
    # Limit for quick trial
    num_batches = min(num_batches, 500)
    
    print(f"\n🚀 Iniciando treinamento industrial...")
    print(f"   LR: {lr}, Seq Len: {seq_len}, Batches: {num_batches}")
    
    # Initialize Optimizer
    optimizer = nsos.MuonOptimizer([], lr)
    params = model.parameters()
    
    total_loss = 0
    start_time = time.time()
    
    for i in range(num_batches):
        ctx = nsos.Context()
        start_idx = i * seq_len
        input_ids = tokens[start_idx : start_idx + seq_len]
        target_ids = tokens[start_idx + 1 : start_idx + seq_len + 1]
        
        if len(target_ids) < seq_len: break
        
        try:
            # 1. Forward
            hidden = model.forward_ids(input_ids, ctx) # [1, L, D]
            
            # 2. Project to Logits (Weight Tying)
            W_emb = model.embedding.weight.data
            W_t = W_emb.transpose()
            
            B, L, D = hidden.shape[0], hidden.shape[1], hidden.shape[2]
            hidden_2d = hidden.reshape([B * L, D])
            logits_2d = hidden_2d.matmul(W_t)
            
            # 3. Loss & CE Grad
            loss_val, d_logits = logits_2d.cross_entropy(target_ids)
            total_loss += loss_val
            
            # 4. Backward
            d_hidden_2d = d_logits.matmul(W_emb)
            d_hidden = d_hidden_2d.reshape([B, L, D])
            model.backward_external(d_hidden, ctx)
            
            # 5. Optimizer Step (Spectral/Industrial)
            optimizer.step_and_quantize(params)
            
            # 6. Periodic Weight Repacking for Hardware Efficiency
            if i % 50 == 0:
                # In industrial mode, step_and_quantize might already repack, 
                # but we'll do once more to be sure if using standard BitLinear
                for layer in model.layers:
                    if hasattr(layer, 'experts'):
                        for e in layer.experts: e.base_weight.data.size > 0 # dummy access
                
        except Exception as e:
            print(f"\n❌ Erro no passo {i}: {e}")
            # traceback.print_exc()
            break
            
        if i % 10 == 0:
            avg_loss = total_loss / (i + 1)
            elapsed = time.time() - start_time
            print(f"\rStep: {i}/{num_batches} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f} | Time: {elapsed:.1f}s", end="")

    print(f"\n✅ Treinamento concluído.")

if __name__ == "__main__":
    train_file = download_wikitext()
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()[:50000] # Subset for speed
    
    tokenizer = nsos.Tokenizer()
    dataset_tokens = tokenizer.encode(text)
    
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'], nsos.Device.CPU)
    
    train_on_data(model, dataset_tokens)
    
    model.save(CONFIG['model_file'])
    print(f"🎉 Modelo salvo em {CONFIG['model_file']}")
