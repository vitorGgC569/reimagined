import json
import os
import torch
import pytest
from torch.utils.data import DataLoader
from oxtacore.converter import OXBConverter
from oxtacore.dataset import OXHDataset

def test_oxh_pipeline_cwd():
    # Setup
    jsonl_file = "test_sample_data.jsonl"
    output_prefix = "test_data"
    num_records = 500

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            record = {
                "text": f"This is sample record number {i}. " * 10,
                "score": 0.5 + (i % 10) / 20.0
            }
            f.write(json.dumps(record) + "\n")

    try:
        # Convert
        converter = OXBConverter()
        converter.convert_jsonl(jsonl_file, output_prefix, shard_size=200)

        # Load
        dataset = OXHDataset(prefix=output_prefix, max_len=128)
        assert len(dataset) == num_records

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        batch = next(iter(dataloader))
        tokens, scores = batch

        assert tokens.shape == (32, 128)
        assert scores.shape == (32,)
        assert tokens.dtype == torch.long
        assert scores.dtype == torch.float32

    finally:
        # Cleanup
        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)
        for f in os.listdir('.'):
            if f.startswith(output_prefix) and (f.endswith('.oxh') or f.endswith('.idx')):
                os.remove(f)
