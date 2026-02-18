import numpy as np
from numba import njit, prange

# V3 Engine: Delta Decoding with Safety Checks

@njit
def fast_delta_decompress(data_buffer):
    """
    Decodifica bits em tokens usando Delta Encoding + Hibridização.
    Versão Segura (Bounds Checking).
    """
    buf_len = len(data_buffer)
    ptr = 0

    # Safety Check 1: Header Size
    if buf_len < 4:
        return np.empty(0, dtype=np.uint32)

    # Read count (uint32)
    token_count = data_buffer[ptr] | (data_buffer[ptr+1] << 8) | (data_buffer[ptr+2] << 16) | (data_buffer[ptr+3] << 24)
    ptr += 4

    if token_count == 0:
        return np.empty(0, dtype=np.uint32)

    output = np.empty(token_count, dtype=np.uint32)

    # Safety Check 2: Base Token existence
    if ptr + 4 > buf_len:
        return np.empty(0, dtype=np.uint32) # Corrupted buffer

    # First token is absolute (uint32)
    last_val = data_buffer[ptr] | (data_buffer[ptr+1] << 8) | (data_buffer[ptr+2] << 16) | (data_buffer[ptr+3] << 24)
    ptr += 4
    output[0] = last_val

    token_idx = 1
    while token_idx < token_count:
        if ptr + 2 > buf_len:
            break # Safety break

        # Read Delta (int16) -> 2 bytes
        val_u16 = data_buffer[ptr] | (data_buffer[ptr+1] << 8)
        ptr += 2

        if val_u16 == 0x8000: # Escape for large delta
            if ptr + 4 > buf_len:
                break

            # Read int32 delta
            delta = data_buffer[ptr] | (data_buffer[ptr+1] << 8) | (data_buffer[ptr+2] << 16) | (data_buffer[ptr+3] << 24)

            # Signed 32-bit conversion
            if delta > 0x7FFFFFFF:
                delta -= 0x100000000

            ptr += 4
        else:
            # Convert uint16 to int16
            delta = val_u16
            if delta > 32767:
                delta -= 65536

        # Apply Delta
        current_val = last_val + delta
        output[token_idx] = current_val
        last_val = current_val
        token_idx += 1

    return output[:token_idx]

def slow_delta_decompress(data_buffer):
    """
    Pure Python version of delta decompressor.
    Safe for environments where Numba causes segfaults.
    """
    # Convert memoryview/bytes to bytearray for easy indexing if needed,
    # but indexing bytes works fine.
    # data_buffer is likely a numpy memmap (behaves like array) or bytes.

    buf_len = len(data_buffer)
    ptr = 0

    if buf_len < 4:
        return np.array([], dtype=np.uint32)

    # Read count
    token_count = int(data_buffer[ptr]) | (int(data_buffer[ptr+1]) << 8) | \
                  (int(data_buffer[ptr+2]) << 16) | (int(data_buffer[ptr+3]) << 24)
    ptr += 4

    if token_count == 0:
        return np.array([], dtype=np.uint32)

    output = np.empty(token_count, dtype=np.uint32)

    if ptr + 4 > buf_len:
        return np.array([], dtype=np.uint32)

    # Base Token
    last_val = int(data_buffer[ptr]) | (int(data_buffer[ptr+1]) << 8) | \
               (int(data_buffer[ptr+2]) << 16) | (int(data_buffer[ptr+3]) << 24)
    ptr += 4
    output[0] = last_val

    token_idx = 1
    while token_idx < token_count:
        if ptr + 2 > buf_len:
            break

        val_u16 = int(data_buffer[ptr]) | (int(data_buffer[ptr+1]) << 8)
        ptr += 2

        if val_u16 == 0x8000:
            if ptr + 4 > buf_len:
                break
            delta = int(data_buffer[ptr]) | (int(data_buffer[ptr+1]) << 8) | \
                    (int(data_buffer[ptr+2]) << 16) | (int(data_buffer[ptr+3]) << 24)
            if delta > 0x7FFFFFFF:
                delta -= 0x100000000
            ptr += 4
        else:
            delta = val_u16
            if delta > 32767:
                delta -= 65536

        last_val = last_val + delta
        output[token_idx] = last_val
        token_idx += 1

    return output[:token_idx]
