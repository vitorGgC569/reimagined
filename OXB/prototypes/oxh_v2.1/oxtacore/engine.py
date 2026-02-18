import numpy as np
from numba import njit

@njit
def fast_hybrid_decompress(data_buffer):
    """Decodifica bits em tokens uint32 usando lógica híbrida."""
    output = np.empty(len(data_buffer), dtype=np.uint32)
    token_ptr = 0
    ptr = 0
    while ptr < len(data_buffer):
        val = data_buffer[ptr] | (data_buffer[ptr + 1] << 8)
        ptr += 2
        if val == 65535: # Sinal de Escape detectado
            actual_val = (data_buffer[ptr]) | (data_buffer[ptr + 1] << 8) | \
                         (data_buffer[ptr + 2] << 16) | (data_buffer[ptr + 3] << 24)
            output[token_ptr] = actual_val
            ptr += 4
        else:
            output[token_ptr] = val
        token_ptr += 1
    return output[:token_ptr]
