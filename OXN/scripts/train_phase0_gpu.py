import sys
import os
import time

# Try importing nsos_ext (Assuming PYTHONPATH is set or in build dir)
try:
    import nsos_ext
except ImportError as e:
    # Fallback to local build directory
    build_path = os.path.join(os.path.dirname(__file__), '../nsos/build')
    sys.path.append(build_path)
    try:
        import nsos_ext
    except ImportError as e2:
        print(f"CRITICAL: Failed to import nsos_ext.\nError 1: {e}\nError 2: {e2}")
        sys.exit(1)

import numpy as np

def test_interop_gpu():
    print("=== 🧱 BLOCO 1: Tensor API (GPU Detection) ===")

    # Check device availability
    try:
        t_gpu = nsos_ext.Tensor.zeros([2, 2], nsos_ext.Device.GPU)
        print(f"Created GPU Tensor: {t_gpu.device}")
    except Exception as e:
        print(f"⚠️ GPU Allocation Failed: {e}")
        print("Assuming CPU-only environment for verification.")
        return

    # If we are here, GPU is active
    print("\n=== 🧱 BLOCO 2: Kernels (Persistent & BitNet) ===")

    # Basic Math
    a = nsos_ext.Tensor.ones([4, 4], nsos_ext.Device.GPU)
    b = nsos_ext.Tensor.ones([4, 4], nsos_ext.Device.GPU)
    c = a.add(b) # Should call launch_add_kernel

    c_cpu = c.cpu()
    print(f"GPU Add Result (First Element): {c_cpu.numpy()[0,0]}")
    if c_cpu.numpy()[0,0] != 2.0:
        print("❌ GPU Add Kernel Failed computation!")
        sys.exit(1)
    else:
        print("✅ GPU Add Kernel Success")

    print("\n=== ✅ GPU SANITY CHECK PASSED ===")

if __name__ == "__main__":
    test_interop_gpu()
