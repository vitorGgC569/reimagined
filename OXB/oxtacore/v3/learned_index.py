import numpy as np
import struct
import os

class LinearLearnedIndex:
    def __init__(self, slope=0.0, intercept=0.0, max_error=0, num_records=0):
        self.slope = slope
        self.intercept = intercept
        self.max_error = max_error
        self.num_records = num_records

    def train(self, offsets):
        """Train using implicit keys 0..N-1"""
        n = len(offsets)
        self.num_records = n
        if n == 0: return

        X = np.arange(n, dtype=np.float64)
        Y = np.array(offsets, dtype=np.float64)
        self.train_explicit(X, Y)

    def train_explicit(self, X, Y):
        """Train using explicit X and Y arrays"""
        n = len(X)
        self.num_records = n if self.num_records == 0 else self.num_records # If training part, keep full num? No, local n.
        if n == 0:
            self.slope = 0; self.intercept = 0; self.max_error = 0; return

        # Simple Linear Regression
        if n > 1:
            A = np.vstack([X, np.ones(n)]).T
            self.slope, self.intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
        else:
            self.slope = 0
            self.intercept = Y[0] # Assuming X[0] is near 0 or handled by intercept logic?
            # If Y = mX + b, and n=1.
            # If we assume global slope 0, then b = Y. But predict(X) will return b.
            # Correct for single point.

        # Calculate Max Error
        predicted = self.slope * X + self.intercept
        errors = np.abs(Y - predicted)
        # We need to store max error as integer (bytes)
        # Using uint64 for safety
        self.max_error = int(np.ceil(np.max(errors)))

    def predict(self, idx):
        return int(self.slope * idx + self.intercept)

    def save(self, filepath, f=None):
        should_close = False
        if f is None:
            f = open(filepath, 'wb')
            should_close = True

        try:
            # Format: [Magic: LIDX][Version: 1][Slope: f64][Intercept: f64][MaxErr: u64][NumRec: u64]
            f.write(struct.pack('<4sIddQQ', b'LIDX', 1, self.slope, self.intercept, self.max_error, self.num_records))
        finally:
            if should_close:
                f.close()

    def load(self, filepath, f=None):
        should_close = False
        if f is None:
            f = open(filepath, 'rb')
            should_close = True

        try:
            header = f.read(40) # 4+4+8+8+8+8
            magic, ver, self.slope, self.intercept, self.max_error, self.num_records = struct.unpack('<4sIddQQ', header)
            if magic != b'LIDX':
                raise ValueError("Invalid Learned Index file")
        finally:
            if should_close:
                f.close()

class RecursiveModelIndex:
    """
    A 2-Stage Recursive Model Index (RMI).
    Integrates V4 research into V3 production.
    """
    def __init__(self, num_leaves=100):
        self.num_leaves = num_leaves
        self.leaves = [LinearLearnedIndex() for _ in range(num_leaves)]
        self.num_records = 0
        self.max_global_error = 0

    def train(self, offsets):
        n = len(offsets)
        self.num_records = n
        if n == 0: return

        keys = np.arange(n, dtype=np.float64)

        # Partition the Index space (0..N) into `num_leaves` equal buckets.
        chunk_size = int(np.ceil(n / self.num_leaves))
        global_max_err = 0

        for i in range(self.num_leaves):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            if start >= end: break

            # Data for this leaf
            X_leaf = keys[start:end]
            Y_leaf = offsets[start:end]

            self.leaves[i].train_explicit(X_leaf, Y_leaf)
            global_max_err = max(global_max_err, self.leaves[i].max_error)

        self.max_global_error = global_max_err

    def predict(self, key):
        if self.num_records == 0: return 0

        # Root inference (Implicit partitioning)
        if key >= self.num_records: key = self.num_records - 1

        chunk_size = int(np.ceil(self.num_records / self.num_leaves))
        leaf_idx = int(key // chunk_size)

        if leaf_idx >= self.num_leaves:
            leaf_idx = self.num_leaves - 1

        return self.leaves[leaf_idx].predict(key)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            # Format: [Magic: RMIDX][Version: 1][NumLeaves: I][NumRec: Q][MaxGlobErr: Q]
            # Followed by N LinearModels
            f.write(struct.pack('<5sIIQQ', b'RMIDX', 1, self.num_leaves, self.num_records, self.max_global_error))

            for leaf in self.leaves:
                leaf.save(None, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            header = f.read(29) # 5+4+4+8+8
            magic, ver, self.num_leaves, self.num_records, self.max_global_error = struct.unpack('<5sIIQQ', header)

            if magic != b'RMIDX':
                raise ValueError("Invalid RMI Index file")

            self.leaves = []
            for _ in range(self.num_leaves):
                leaf = LinearLearnedIndex()
                leaf.load(None, f)
                self.leaves.append(leaf)

def train_and_save_index(offsets, filepath, model_type='linear'):
    if model_type == 'rmi':
        model = RecursiveModelIndex(num_leaves=100) # Default to 100 partitions
    else:
        model = LinearLearnedIndex()

    model.train(offsets)
    model.save(filepath)
    return model
