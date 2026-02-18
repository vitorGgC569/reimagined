import json
import os
import torch
import pytest
import gc
import numpy as np
from torch.utils.data import DataLoader
from oxtacore.v3.converter import OXBConverterV3
from oxtacore.v3.dataset import OXHDatasetV3

def test_oxh_v3_pipeline_cwd():
    # Setup
    jsonl_file = "test_v3_data.jsonl"
    output_prefix = "test_v3_data"
    num_records = 500

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            record = {
                "text": f"This is sample record number {i}. " * 10,
                "score": 0.5 + (i % 10) / 20.0
            }
            f.write(json.dumps(record) + "\n")

    dataset = None
    dataloader = None

    try:
        # Convert V3
        converter = OXBConverterV3()
        converter.convert_jsonl(jsonl_file, output_prefix, shard_size=200)

        # Load V3
        dataset = OXHDatasetV3(prefix=output_prefix, max_len=128)
        assert len(dataset) == num_records

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        batch = next(iter(dataloader))
        tokens, scores = batch

        assert tokens.shape == (32, 128)
        assert scores.shape == (32,)
        assert tokens.dtype == torch.long
        assert scores.dtype == torch.float32

    finally:
        # Explicit Cleanup for Windows File Locking
        if dataset and hasattr(dataset, 'shards'):
            for shard in dataset.shards:
                if 'd' in shard and isinstance(shard['d'], np.memmap):
                    if hasattr(shard['d'], '_mmap'):
                        shard['d']._mmap.close()
                    del shard['d']

        del dataset
        del dataloader
        gc.collect()

        # Cleanup Files
        if os.path.exists(jsonl_file):
            try:
                os.remove(jsonl_file)
            except OSError:
                pass

        for f in os.listdir('.'):
            if f.startswith(output_prefix) and (f.endswith('.ox3') or f.endswith('.lin')):
                try:
                    os.remove(f)
                except OSError:
                    pass
