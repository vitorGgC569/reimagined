import os
import numpy as np
import struct
import torch
from torch.utils.data import Dataset
from .engine import fast_hybrid_decompress

class OXHDataset(Dataset):
    def __init__(self, prefix, max_len=512):
        self.max_len = max_len
        self.shards = []
        self.cum_sizes = [0]
        paths = sorted([f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.oxh')])

        for p in paths:
            self.shards.append({'d': np.memmap(p, dtype='uint8', mode='r'), 'i': np.fromfile(p + ".idx", dtype=np.uint64)})
            self.cum_sizes.append(self.cum_sizes[-1] + len(self.shards[-1]['i']))

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        s_idx = np.searchsorted(self.cum_sizes, idx, side='right') - 1
        l_idx = idx - self.cum_sizes[s_idx]
        shard = self.shards[s_idx]
        off = int(shard['i'][l_idx])

        score_bytes = shard['d'][off : off+4].tobytes()
        score = struct.unpack('<f', score_bytes)[0]

        size_bytes = shard['d'][off+4 : off+8].tobytes()
        size = struct.unpack('<I', size_bytes)[0]

        tokens = fast_hybrid_decompress(shard['d'][off+8 : off+8+size])

        # Optimization: Return numpy array directly and pad/truncate efficiently
        if len(tokens) > self.max_len:
            # Note: fast_hybrid_decompress returns uint32. PyTorch expects int64 (long) for embeddings usually.
            return tokens[:self.max_len].astype(np.int64), np.float32(score)

        res = np.zeros(self.max_len, dtype=np.int64)
        res[:len(tokens)] = tokens.astype(np.int64)

        # Return numpy array and float32. DataLoader will collate them into Tensors.
        return res, np.float32(score)
