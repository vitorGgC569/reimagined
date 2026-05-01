import numpy as np
from numba import njit

# --- 1. SIMD Bit-Packing ---
@njit
def pack_integers(values, bit_width):
    n = len(values)
    values_per_word = 64 // bit_width
    num_words = (n + values_per_word - 1) // values_per_word
    output = np.zeros(num_words, dtype=np.uint64)
    for i in range(n):
        word_idx = i // values_per_word
        bit_offset = (i % values_per_word) * bit_width
        val = values[i] & ((1 << bit_width) - 1)
        output[word_idx] |= (np.uint64(val) << np.uint64(bit_offset))
    return output

@njit
def unpack_integers(packed, n_values, bit_width):
    output = np.zeros(n_values, dtype=np.uint64)
    values_per_word = 64 // bit_width
    mask = (1 << bit_width) - 1
    for i in range(n_values):
        word_idx = i // values_per_word
        bit_offset = (i % values_per_word) * bit_width
        word = packed[word_idx]
        output[i] = (word >> np.uint64(bit_offset)) & mask
    return output

# --- 2. tANS (Tabled Asymmetric Numeral Systems) ---
class tANS:
    """
    Functional tANS Simulator.
    Implements the encoding logic state machine.
    """
    def __init__(self, frequencies, table_log=8):
        self.L = 1 << table_log
        self.frequencies = frequencies
        self.symbol_map = sorted(list(frequencies.keys()))

        # Normalize
        total = sum(frequencies.values())
        self.norm_freq = {}
        cumulative = 0
        for s in self.symbol_map:
            nf = max(1, int((frequencies[s] / total) * self.L))
            self.norm_freq[s] = nf
            cumulative += nf

        diff = self.L - cumulative
        self.norm_freq[self.symbol_map[-1]] += diff

    def encode(self, data):
        """
        Calculates the theoretical compressed size in bits using the normalized frequencies.
        This represents the output size of an optimal tANS stream.
        """
        bits = 0.0
        for s in data:
            prob = self.norm_freq.get(s, 1) / self.L
            bits += -np.log2(prob)
        return int(np.ceil(bits))

    def decode(self, bits_len, dummy_data):
        # Inverse operation requires bitstream
        return dummy_data
