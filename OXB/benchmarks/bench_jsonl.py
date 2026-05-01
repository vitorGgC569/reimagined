import time
import json
import torch
import tiktoken
import numpy as np
from torch.utils.data import DataLoader, Dataset
from benchmarks.common import NUM_SAMPLES, JSONL_FILE, BATCH_SIZE, NUM_WORKERS, MAX_LEN

class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer_name="cl100k_base", max_len=MAX_LEN):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        tokens = self.enc.encode(obj["text"])

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[:len(tokens)] = tokens

        return torch.from_numpy(padded), torch.tensor(obj["score"], dtype=torch.float32)

def benchmark():
    print(f"\n--- Benchmark: JSONL (Runtime Tokenization) ---")

    t0 = time.time()
    dataset = JsonlDataset(JSONL_FILE)
    t1 = time.time()
    print(f"Dataset Init (Load+Parse JSON to RAM): {t1-t0:.4f}s")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    start_time = time.time()
    total = 0
    for batch in loader:
        total += batch[0].shape[0]
    end_time = time.time()

    duration = end_time - start_time
    throughput = total / duration

    print(f"Processed {total} samples in {duration:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    return throughput

if __name__ == "__main__":
    benchmark()
