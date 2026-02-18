import sys
import os
import time
import random
import urllib.request
import re
import subprocess
import math

# Setup environment
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Release"))
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
    if os.name == 'nt':
        os.add_dll_directory(build_dir)

# CUDA DLLs for Windows
if os.name == 'nt':
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)

import nsos_ext as nsos
import numpy as np
from tqdm import tqdm

# CONFIG
CONFIG = {
    "layers": 4, 
    "dim": 128,
    "vocab": 256, 
    "seq_len": 32,
    'steps': 50, # Fast trial
    'lr': 0.0001,
    'model_file': 'oxn_bpe_test.bin',
    "dataset_url": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "data_dir": "wikitext_data",
}

def download_wikitext():
    if not os.path.exists(CONFIG['data_dir']):
        os.makedirs(CONFIG['data_dir'])
    train_path = os.path.join(CONFIG['data_dir'], "wiki.train.raw")
    if not os.path.exists(train_path):
        print(f"📥 Downloading dataset...")
        subprocess.run(["curl", "-k", "-L", "-o", train_path, CONFIG['dataset_url']], check=True)
    return train_path

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return re.sub(r'\s+', ' ', text)

def train():
    print("--- NSOS WikiText Training with NEW BPE Tokenizer ---")
    train_file = download_wikitext()
    text = load_text(train_file)
    
    # 1. Initialize the NEW Tokenizer
    print("🔋 Initializing C++ Tokenizer...")
    tokenizer = nsos.Tokenizer()
    tokenizer.add_special_tokens(["<|endoftext|>", "<|pad|>"])
    
    # 2. Tokenize using the optimized C++ engine
    print(f"🔠 Tokenizing {len(text[:100000])} characters (subset for trial)...")
    # For a full test we could tokenize all, but for verification let's take a large chunk
    tokens = tokenizer.encode(text[:100000]) 
    print(f"✅ Tokenization complete. Total tokens: {len(tokens)}")

    # Initialize Model components
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'])
    
    params = list(model.parameters())
    
    seq_len = CONFIG['seq_len']
    lr = CONFIG['lr']
    
    total_loss = 0
    pbar = tqdm(total=CONFIG['steps'], desc="Training Trial")
    
    for i in range(CONFIG['steps']):
        start_idx = random.randint(0, len(tokens) - seq_len - 1)
        input_ids = tokens[start_idx : start_idx + seq_len]
        target_ids = tokens[start_idx + 1 : start_idx + seq_len + 1]
        
        ctx = nsos.Context()
        
        # Forward (now returns logits directly)
        logits = model.forward_ids(input_ids, ctx)
        logits_2d = logits.reshape([seq_len, CONFIG['vocab']])
        
        # Loss
        loss_val, d_logits = logits_2d.cross_entropy(target_ids)
        total_loss += loss_val
        
        # Backward
        model.backward_external(d_logits.reshape([1, seq_len, CONFIG['vocab']]), ctx)
        
        # Update
        grads = [p.grad for p in params if p.grad is not None and p.grad.size > 0]
        if grads:
            nsos.Tensor.clip_grad_norm_(grads, 1.0)
            
        for p in params:
            if p.grad is not None and p.grad.size > 0:
                p.data.copy_from(p.data.sub(p.grad.mul(lr)))
                p.zero_grad()
                
        pbar.update(1)
        pbar.set_postfix({'loss': f'{float(loss_val):.4f}'})

    pbar.close()
    print(f"\n✅ Trial Finished. Avg Loss: {total_loss/CONFIG['steps']:.4f}")
    
    # Verify Decode works
    test_decode = tokenizer.decode(tokens[:10])
    print(f"📝 Verification Decode (First 10 tokens): '{test_decode}'")

if __name__ == "__main__":
    train()
