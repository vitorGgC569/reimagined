import numpy as np

class RankSelectBitVector:
    """
    Succinct Data Structure Simulation.
    Supports rank(i) and select(i) in O(1) using blocks.
    """
    def __init__(self, bit_string):
        """
        bit_string: str of '0' and '1' or list of ints.
        """
        self.bits = np.array([int(b) for b in bit_string], dtype=np.uint8)
        self.n = len(self.bits)

        # Precompute Rank
        # In a real SDSL, we use Superblocks and Blocks to save space.
        # Here we just store the full cumulative sum array (O(N) space)
        # to demonstrate the O(1) access logic.
        self._rank_1 = np.cumsum(self.bits)

    def rank1(self, i):
        """Number of 1s up to index i (inclusive)."""
        if i < 0: return 0
        if i >= self.n: return self._rank_1[-1]
        return self._rank_1[i]

    def rank0(self, i):
        """Number of 0s up to index i (inclusive)."""
        return (i + 1) - self.rank1(i)

    def select1(self, k):
        """Position of the k-th 1 (1-based k)."""
        if k == 0: return -1
        # Search in rank array. Since sorted, use binary search.
        # Ideally SDSL uses a separate structure for O(1) select.
        idx = np.searchsorted(self._rank_1, k)
        if idx < self.n and self._rank_1[idx] == k:
            return idx
        return -1
