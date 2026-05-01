import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import DataLoader
from oxtacore.v3.converter import OXBConverterV3
from oxtacore.v3.dataset import OXHDatasetV3
from oxtacore.model import OxtaN, Config
import numpy as np

VOCAB_SIZE = 100277
DIM = 128
LAYERS = 2
HEADS = 4
CTX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 2

JSONL_FILE = "tribute.jsonl"
OXH_PREFIX = "tribute_data"
MODEL_PATH = "oxta_devoto.pt"

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Treinando Oxta-Devoto em: {device}")

    # 1. Converter
    # Clean previous if exists
    if os.path.exists(OXH_PREFIX + "_part_000.ox3"):
        for f in os.listdir('.'):
            if f.startswith(OXH_PREFIX):
                try:
                    os.remove(f)
                except OSError:
                    pass

    print("Convertendo para OXH V3.1 (Format: Delta Encoded .ox3)...")
    converter = OXBConverterV3()
    converter.convert_jsonl(JSONL_FILE, OXH_PREFIX, shard_size=500)

    # 2. Carregar
    print("Carregando Dataset OXH V3.1...")
    dataset = OXHDatasetV3(OXH_PREFIX, max_len=CTX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Modelo
    config = Config(vocab_size=VOCAB_SIZE, n_embd=DIM, n_head=HEADS, n_layer=LAYERS, block_size=CTX_LEN)
    model = OxtaN(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
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

            weighted_loss = (loss_per_sample * scores).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {weighted_loss.item():.4f}")

        print(f"--- Epoch {epoch} Final Loss: {total_loss / len(loader):.4f} ---")

    print("Treinamento concluído. Salvando modelo...")
    torch.save(model.state_dict(), MODEL_PATH)

    # Cleanup to avoid Windows locking
    if hasattr(dataset, 'shards'):
        for shard in dataset.shards:
            if 'd' in shard and isinstance(shard['d'], np.memmap):
                if hasattr(shard['d'], '_mmap'):
                    shard['d']._mmap.close()
                del shard['d']
    del dataset
    del loader
    gc.collect()

if __name__ == "__main__":
    train()
