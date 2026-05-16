import os
import sys

# Add CUDA DLLs to search path on Windows
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if sys.platform == "win32" and os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)

import nsos_ext as nsos
import numpy as np

def test_gpu():
    print("Testing NSOS CUDA Support (Simplified)...")
    try:
        device = nsos.Device.GPU
        print(f"Device: {device}")
        
        # Test 1: BitLinear on GPU
        print("Creating BitLinear on GPU...")
        linear = nsos.BitLinear(128, 128)
        linear.to(device)
        print("BitLinear created and moved to GPU.")
        
        # Test 2: Forward
        x = nsos.Tensor([16, 128], device, 0.5)
        print("Starting Forward (BitLinear)...")
        out = linear.forward(x)
        print(f"Forward OK. Output norm: {out.norm()}")
        
        print("✅ NSOS CUDA BITLINEAR TEST PASSED!")
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")

if __name__ == "__main__":
    test_gpu()
