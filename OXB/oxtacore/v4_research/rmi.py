import numpy as np
from oxtacore.v3.learned_index import LinearLearnedIndex

class RecursiveModelIndex:
    """
    A 2-Stage Recursive Model Index (RMI).

    Structure:
    - Root Model: Predicts which "Leaf Model" to use.
    - Leaf Models: Array of Linear Models, each covering a partition of the data.

    Why?
    A single Linear Model (V3.1) fails if data has non-linear distribution (e.g. gaps, clusters).
    RMI approximates the CDF using piecewise linear functions.
    """
    def __init__(self, num_leaves=10):
        self.num_leaves = num_leaves
        self.root = LinearLearnedIndex()
        self.leaves = [LinearLearnedIndex() for _ in range(num_leaves)]
        self.max_global_error = 0

    def train(self, keys, offsets):
        """
        keys: Inputs (e.g., Sample IDs 0, 1, 2...). In OXH, usually just 0..N-1.
              If keys are 0..N-1, RMI is trivial (just math).
              BUT, usually RMI maps Key -> Position.
              In OXH, we map SampleID (Index) -> FileOffset (Value).
              If offsets are non-linear (compressed variable length), RMI helps.
        offsets: The target values (File positions).
        """
        n = len(keys)
        if n == 0: return

        # 1. Train Root Model
        # Root maps Key -> Leaf Index (0 to num_leaves-1)
        # We want to partition the keys space evenly.
        # Simple approach: Root maps key to a float 0..num_leaves

        # Training Root:
        # X = keys
        # Y = scaled leaf index (e.g. key / max_key * num_leaves)
        # Actually, for SampleID (0..N), the distribution of KEYS is uniform.
        # The distribution of OFFSETS is what varies.
        # Wait. RMI usually maps Key -> Position in sorted array.
        # Here we map Index -> Offset.
        # If record sizes vary wildly, Offset(Index) is non-linear.

        # Let's use Root to approximate the CDF of offsets? No.
        # We use Root to pick a model that approximates the local slope of Index->Offset.

        # Simplest RMI for Index -> Offset:
        # Partition the Index space (0..N) into `num_leaves` equal buckets.
        # Train a Linear Model for each bucket.
        # Root model is just: leaf_idx = floor(key / N * num_leaves).
        # This is "Radix" root.

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
        # Root is implicit in this simple partitioning scheme
        self.n = n

    def predict(self, key):
        # 1. Root inference (Implicit partitioning)
        # Which leaf?
        if self.n == 0: return 0

        # leaf_idx = floor(key / n * num_leaves)
        # Need to handle key >= n bounds
        if key >= self.n: key = self.n - 1

        chunk_size = int(np.ceil(self.n / self.num_leaves))
        leaf_idx = int(key // chunk_size)

        if leaf_idx >= self.num_leaves:
            leaf_idx = self.num_leaves - 1

        # 2. Leaf inference
        return self.leaves[leaf_idx].predict(key)

class EnhancedLinearIndex(LinearLearnedIndex):
    """Extension to allow training with explicit X, Y arrays."""
    def train_explicit(self, X, Y):
        n = len(X)
        if n == 0:
            self.slope = 0; self.intercept = 0; self.max_error = 0; return

        A = np.vstack([X, np.ones(n)]).T
        self.slope, self.intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

        predicted = self.slope * X + self.intercept
        errors = np.abs(Y - predicted)
        self.max_error = int(np.ceil(np.max(errors)))

# Patch the leaf class
LinearLearnedIndex.train_explicit = EnhancedLinearIndex.train_explicit
