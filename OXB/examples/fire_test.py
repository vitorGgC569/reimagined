import time
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import DataLoader
from oxtacore.v3.converter import OXBConverterV3
from oxtacore.v3.dataset import OXHDatasetV3
from oxtacore.model import OxtaN, Config

# Fire Test Configuration
VOCAB_SIZE = 1000
DIM = 128
LAYERS = 2
HEADS = 4
CTX_LEN = 64
NUM_SAMPLES = 2000
BATCH_SIZE = 16
EPOCHS = 2

JSONL_FILE = "fire_data.jsonl"
OXH_PREFIX = "fire_data"

def generate_fire_data():
    print("Generating Fire Test Data...")
    with open(JSONL_FILE, 'w') as f:
        for i in range(NUM_SAMPLES):
            if i % 2 == 0:
                # Good sample
                text = "The quick brown fox jumps over the lazy dog." * 3
                score = 1.0
            else:
                # Bad sample (random noise)
                text = "x y z a b c " * 5
                score = 0.1
            f.write(json.dumps({"text": text, "score": score}) + "\n")

def fire_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Fire Test on: {device}")

    if os.path.exists(OXH_PREFIX + "_part_000.ox3"):
        for f in os.listdir('.'):
             if f.startswith(OXH_PREFIX):
                 try:
                     os.remove(f)
                 except OSError:
                     pass

    if os.path.exists(JSONL_FILE): os.remove(JSONL_FILE)
    generate_fire_data()

    print("\n--- Converting to OXH V3.1 (Delta Encoded + Learned Index) ---")
    converter = OXBConverterV3()
    converter.convert_jsonl(JSONL_FILE, OXH_PREFIX, shard_size=1000)

    print("\n--- Loading Dataset (OXH V3.1) ---")
    dataset = OXHDatasetV3(OXH_PREFIX, max_len=CTX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = Config(vocab_size=VOCAB_SIZE, n_embd=DIM, n_head=HEADS, n_layer=LAYERS, block_size=CTX_LEN)
    model = OxtaN(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("\n--- Starting Oxta-N Training ---")
    model.train()

    weighted_loss = None
    for epoch in range(EPOCHS):
        for i, (tokens, scores) in enumerate(loader):
            tokens, scores = tokens.to(device), scores.to(device)
            tokens = torch.clamp(tokens, 0, VOCAB_SIZE-1)

            idx = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(idx)

            B, T, V = logits.shape
            loss_per_token = criterion(logits.reshape(-1, V), targets.reshape(-1))
            loss_per_sample = loss_per_token.view(B, T).mean(dim=1)

            # Weighted Loss
            weighted_loss = (loss_per_sample * scores).mean()

            weighted_loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {weighted_loss.item():.4f}")

    print("\n--- Fire Test Results ---")
    if weighted_loss is not None:
        print(f"Final Loss: {weighted_loss.item():.4f}")
    print("Oxta-N trained successfully.")

    # Cleanup with explicit close
    print("Cleaning up...")

    # Close memmaps
    if hasattr(dataset, 'shards'):
        for shard in dataset.shards:
            if 'd' in shard and isinstance(shard['d'], np.memmap):
                if hasattr(shard['d'], '_mmap'):
                    shard['d']._mmap.close()
                del shard['d']

    del dataset
    del loader
    gc.collect()

    if os.path.exists(JSONL_FILE): os.remove(JSONL_FILE)
    for f in os.listdir('.'):
        if f.startswith(OXH_PREFIX):
            try:
                os.remove(f)
            except OSError as e:
                print(f"Warning: Could not remove {f}: {e}")

if __name__ == "__main__":
    import numpy as np # Needed for cleanup check logic
    fire_test()
