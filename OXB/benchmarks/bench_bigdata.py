import time
import os
import json
import torch
import tiktoken
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from oxtacore.v3.converter import OXBConverterV3
from oxtacore.v3.dataset import OXHDatasetV3

# Configuration
NUM_SAMPLES = 20000 # Reduced to avoid sandbox git limit
BATCH_SIZE = 64
NUM_WORKERS = 2
MAX_LEN = 128
DATA_DIR = "bench_bigdata"
JSONL_FILE = f"{DATA_DIR}/bigdata.jsonl"
OXH_PREFIX = f"{DATA_DIR}/bigdata_oxh"

def generate_bigdata():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Generating {NUM_SAMPLES} samples for Big Data Test...")
    with open(JSONL_FILE, 'w', encoding='utf-8') as f:
        for i in range(NUM_SAMPLES):
            text = "The quick brown fox jumps over the lazy dog. " * 5
            score = 1.0
            f.write(json.dumps({"text": text, "score": score}) + "\n")

# --- Realistic Big Data Loaders (No RAM Caching) ---

class JsonlStreamingDataset(Dataset):
    def __init__(self, path, tokenizer_name="cl100k_base", max_len=MAX_LEN):
        self.path = path
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.max_len = max_len
        self.offsets = [0]

        # Build Index (Required for random access shuffle)
        print("Indexing JSONL offsets...")
        t0 = time.time()
        with open(path, 'rb') as f:
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop()
        print(f"JSONL Indexing took {time.time()-t0:.2f}s")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        # Simulate Big Data: Open, Seek, Read, Parse, Close (or keep open per worker)
        # Standard python 'open' is slow if done every time.
        # We assume smart worker usage, but let's do the standard safe way: open/seek.

        with open(self.path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline()

        obj = json.loads(line)
        tokens = self.enc.encode(obj["text"])

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[:len(tokens)] = tokens

        return torch.from_numpy(padded), torch.tensor(obj["score"], dtype=torch.float32)

def benchmark():
    print(f"\n=== BIG DATA BENCHMARK ({NUM_SAMPLES} samples) ===")
    print("Scenario: Dataset > RAM (Simulated by streaming from disk)")

    # 1. Setup Data
    if os.path.exists(DATA_DIR):
        import shutil
        shutil.rmtree(DATA_DIR)
    generate_bigdata()

    # 2. JSONL Streaming Benchmark
    print("\n--- JSONL Streaming (Seek + Parse + Tokenize) ---")
    dataset_json = JsonlStreamingDataset(JSONL_FILE)
    loader_json = DataLoader(dataset_json, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    t0 = time.time()
    count = 0
    for batch in loader_json:
        count += batch[0].shape[0]
    t1 = time.time()

    json_tps = NUM_SAMPLES / (t1-t0)
    print(f"JSONL Time: {t1-t0:.2f}s")
    print(f"JSONL Throughput: {json_tps:.2f} samples/s")

    # 3. OXH Benchmark
    print("\n--- OXH V3.1 (Mmap + Learned Index + Delta Safe) ---")

    print("Converting...")
    tc0 = time.time()
    converter = OXBConverterV3()
    converter.convert_jsonl(JSONL_FILE, OXH_PREFIX, shard_size=50000)
    print(f"Conversion: {time.time()-tc0:.2f}s")

    cwd = os.getcwd()
    os.chdir(DATA_DIR)
    try:
        dataset_oxh = OXHDatasetV3("bigdata_oxh", max_len=MAX_LEN)
        loader_oxh = DataLoader(dataset_oxh, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        t0 = time.time()
        count = 0
        for batch in loader_oxh:
            count += batch[0].shape[0]
        t1 = time.time()

        oxh_tps = NUM_SAMPLES / (t1-t0)
        print(f"OXH Time: {t1-t0:.2f}s")
        print(f"OXH Throughput: {oxh_tps:.2f} samples/s")

        # Cleanup Handlers
        for shard in dataset_oxh.shards:
            if 'd' in shard and hasattr(shard['d'], '_mmap'):
                shard['d']._mmap.close()
            del shard['d']
        del dataset_oxh
        del loader_oxh
        gc.collect()

    finally:
        os.chdir(cwd)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"JSONL Streaming: {json_tps:.2f} samples/s")
    print(f"OXH V3.1 (Safe): {oxh_tps:.2f} samples/s")
    print(f"Speedup: {oxh_tps / json_tps:.2f}x")

if __name__ == "__main__":
    benchmark()
