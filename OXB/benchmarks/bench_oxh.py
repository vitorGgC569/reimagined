import time
import os
import torch
import shutil
import gc
import numpy as np
from torch.utils.data import DataLoader
from benchmarks.common import NUM_SAMPLES, JSONL_FILE, OXH_PREFIX, BATCH_SIZE, NUM_WORKERS, MAX_LEN
from oxtacore.v3.converter import OXBConverterV3
from oxtacore.v3.dataset import OXHDatasetV3

def benchmark():
    print(f"\n--- Benchmark: OXH V3.1 (Binary Delta + Learned Index) ---")

    # 0. Cleanup
    if os.path.exists(os.path.dirname(OXH_PREFIX)):
        # Clear OXH files
        for f in os.listdir(os.path.dirname(OXH_PREFIX)):
            if f.startswith("data_oxh"):
                try:
                    os.remove(os.path.join(os.path.dirname(OXH_PREFIX), f))
                except: pass

    # 1. Conversion
    print("Converting JSONL to OXH...")
    t0 = time.time()
    converter = OXBConverterV3()
    converter.convert_jsonl(JSONL_FILE, OXH_PREFIX, shard_size=20000)
    t1 = time.time()
    print(f"Conversion Time: {t1-t0:.4f}s")

    # 2. Setup Dataset
    cwd = os.getcwd()
    target_dir = os.path.dirname(OXH_PREFIX)
    file_prefix = os.path.basename(OXH_PREFIX)

    os.chdir(target_dir)
    try:
        t0 = time.time()
        dataset = OXHDatasetV3(file_prefix, max_len=MAX_LEN)
        t1 = time.time()
        print(f"Dataset Init (Open Memmaps + Load Indexes): {t1-t0:.4f}s")

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # 3. Loop
        start_time = time.time()
        total = 0
        for batch in loader:
            total += batch[0].shape[0]
        end_time = time.time()

        duration = end_time - start_time
        throughput = total / duration

        print(f"Processed {total} samples in {duration:.4f}s")
        print(f"Throughput: {throughput:.2f} samples/s")

        # Cleanup
        for shard in dataset.shards:
            if 'd' in shard and hasattr(shard['d'], '_mmap'):
                shard['d']._mmap.close()
            del shard['d']
        del dataset
        del loader
        gc.collect()

        return throughput

    finally:
        os.chdir(cwd)

if __name__ == "__main__":
    benchmark()
