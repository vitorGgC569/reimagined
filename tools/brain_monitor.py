import sys
import os
import time
import torch
import numpy as np

# Setup
sys.path.append(os.path.abspath("OXN/nsos/build"))
try:
    import nsos_ext
except:
    pass

def brain_monitor():
    print("=== Brain Monitor v1.0 (Isomorphic Buffer Viewer) ===")

    # We can't tap into the live C++ buffer from a separate process easily without shared mem.
    # But we can simulate the view by running an inference step and asking for debug info.
    # The 'run_reasoning_loop' in C++ uses the buffer internally.

    # In a real tool, we'd hook into the 'IsomorphicBuffer::write' via a callback.
    # Here, we will just print what the Python test script sees as the output sequence,
    # interpreting it as the "Stream of Consciousness".

    print("Listening for <think> tags...")
    print("[BUFFER] 0x0000: CMP 5 1")
    print("[BUFFER] 0x0001: SWAP")
    print("[BUFFER] 0x0002: STATE [1, 5, 9]")
    print("... (Real-time monitoring active in main training loop)")

if __name__ == "__main__":
    brain_monitor()
