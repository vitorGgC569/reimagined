import os
import sys
import numpy as np

# Robust DLL and Path setup for Windows
script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
build_release = os.path.join(root_dir, "build/Release")

sys.path.append(root_dir)
sys.path.append(build_release)

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    if os.path.exists(build_release):
        os.add_dll_directory(os.path.abspath(build_release))
    # Standard CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    ]
    for p in cuda_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

import nsos_ext

def verify_cuda():
    print("=== NSOS CUDA VERIFICATION ===")
    
    # Check if Device.GPU exists
    try:
        dev_gpu = nsos_ext.Device.GPU
        print("[1] Device.GPU is available in bindings.")
    except AttributeError:
        print("[FAIL] Device.GPU NOT found in bindings.")
        return

    # Try creating a GPU tensor
    try:
        print("[2] Attempting to create GPU tensor [1024]...")
        t_gpu = nsos_ext.Tensor.random([1024], dev_gpu)
        print(f"PASS: Tensor created on {t_gpu.device}")
        
        # Test basic op (if available in bindings)
        print("[3] Testing GPU memory move (to CPU)...")
        t_cpu = t_gpu.to(nsos_ext.Device.CPU)
        print(f"PASS: Tensor moved to {t_cpu.device}")
        
    except Exception as e:
        print(f"[FAIL] CUDA Operation error: {e}")
        return

    print("=== VERIFICATION PASSED: CUDA KERNEL ACTIVE ===")

if __name__ == "__main__":
    verify_cuda()
