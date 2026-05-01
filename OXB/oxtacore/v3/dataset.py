import os
import numpy as np
import struct
import torch
from torch.utils.data import Dataset
from .engine import fast_delta_decompress, slow_delta_decompress
from .learned_index import LinearLearnedIndex

# Environment Flag to force Safe Mode
FORCE_SAFE_MODE = True

class OXHDatasetV3(Dataset):
    def __init__(self, prefix, max_len=512):
        self.max_len = max_len
        self.shards = []
        self.cum_sizes = [0]
        # Find .ox3 files and their paired .lin files
        paths = sorted([f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.ox3')])

        for p in paths:
            # Load Learned Index
            lin_path = p + ".lin"
            lidx = LinearLearnedIndex()
            lidx.load(lin_path)

            num_records = lidx.num_records

            self.shards.append({
                'd': np.memmap(p, dtype='uint8', mode='r'),
                'idx': lidx
            })
            self.cum_sizes.append(self.cum_sizes[-1] + num_records)

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        s_idx = np.searchsorted(self.cum_sizes, idx, side='right') - 1
        l_idx = idx - self.cum_sizes[s_idx]
        shard = self.shards[s_idx]
        mem = shard['d']
        model = shard['idx']

        # 1. Predict Offset
        pred_off = model.predict(l_idx)

        # 2. Local Search
        start_search = max(0, pred_off - model.max_error - 16)
        end_search = min(len(mem), pred_off + model.max_error + 16)

        # Optimization: Don't read huge chunk.
        # Just assume prediction is good for now or scan small window.
        # Since we don't have C++ `find`, we trust `pred_off` lands on a marker or close.
        # We need to find `0xDEADBEEF`

        marker = b'\xef\xbe\xad\xde'

        # Simple Scan around prediction
        actual_off = pred_off

        # Check spot on
        if pred_off + 4 <= len(mem) and mem[pred_off:pred_off+4].tobytes() == marker:
            actual_off = pred_off
        else:
            # Linear scan outwards
            found = False
            for drift in range(1, model.max_error + 32):
                p_up = pred_off + drift
                if p_up + 4 <= len(mem) and mem[p_up:p_up+4].tobytes() == marker:
                    actual_off = p_up
                    found = True
                    break
                p_down = pred_off - drift
                if p_down >= 0 and p_down + 4 <= len(mem) and mem[p_down:p_down+4].tobytes() == marker:
                    actual_off = p_down
                    found = True
                    break
            if not found:
                # Fallback: Just try to read at prediction, might be garbage but won't crash safely
                actual_off = pred_off

        off = actual_off + 4 # Skip marker

        # Ensure we have enough bytes for header
        if off + 8 > len(mem):
             return np.zeros(self.max_len, dtype=np.int64), np.float32(0.0)

        score_bytes = mem[off : off+4].tobytes()
        score = struct.unpack('<f', score_bytes)[0]

        size_bytes = mem[off+4 : off+8].tobytes()
        size = struct.unpack('<I', size_bytes)[0]

        payload = mem[off+8 : off+8+size]

        # USE SAFE DECOMPRESSOR
        if FORCE_SAFE_MODE:
            # Convert memmap slice to bytes explicitly for pure python processing
            # or pass memmap slice (it works as buffer)
            tokens = slow_delta_decompress(payload)
        else:
            tokens = fast_delta_decompress(payload)

        if len(tokens) > self.max_len:
            return tokens[:self.max_len].astype(np.int64), np.float32(score)

        res = np.zeros(self.max_len, dtype=np.int64)
        res[:len(tokens)] = tokens.astype(np.int64)

        return res, np.float32(score)
