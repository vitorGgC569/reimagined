import sys
import os
import time

# FIX: Add CUDA bin to PATH/DLL Directory for Windows
if os.name == 'nt':
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']

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

def test_interop_cpu():
    print("=== 🧱 BLOCO 1: Tensor API (CPU) ===")

    # 1. Basic Creation
    t = nsos_ext.Tensor([4, 4], nsos_ext.Device.CPU)
    print(f"Created Tensor: Shape={t.shape}, Device={t.device}")
    assert t.device == nsos_ext.Device.CPU

    # 2. Numpy Interop
    np_arr = t.numpy()
    print(f"Numpy View: Shape={np_arr.shape}, Dtype={np_arr.dtype}")
    assert np_arr.shape == (4, 4)

    # 3. Operations (CPU)
    a = nsos_ext.Tensor.random([4, 4], nsos_ext.Device.CPU)
    b = nsos_ext.Tensor.ones([4, 4], nsos_ext.Device.CPU)
    c = a.matmul(b)
    print(f"Matmul Result: Shape={c.shape}")
    # Fix: Compare list to list or cast tuple
    assert list(c.shape) == [4, 4]

    print("\n=== 🧱 BLOCO 2 & 5: Model Forward & Segfault Check ===")

    vocab_size = 128
    d_model = 16
    model = nsos_ext.JambaModel(2, d_model, vocab_size, nsos_ext.Device.CPU)
    ctx = nsos_ext.Context()

    input_ids = [1, 2, 3, 4]
    print(f"Forwarding IDs: {input_ids}")

    # This was causing segfault
    try:
        out = model.forward_ids(input_ids, ctx)
        print(f"Forward Output: Shape={out.shape}")
        # Fix assertion here too just in case
        assert list(out.shape) == [1, 4, d_model]
    except Exception as e:
        print(f"FORWARD FAILED: {e}")
        sys.exit(1)

    print("\n=== 🕵️ INVESTIGAÇÃO: MoE Bypassing ===")
    # Vamos verificar se os experts recebem gradiente
    print("Checking Expert Gradients...")
    # JambaModel layer 1 should be MoE (i%2 == 1)
    params = model.parameters()
    # Find router gate weights
    # Assuming standard names (need to verify param names exposed)
    
    found_moe = False
    for p in params:
        if "expert" in p.name:
            found_moe = True
            print(f"Found Expert Param: {p.name} | Grad Sum: {p.grad.norm()}")
    
    if not found_moe:
        print("WARNING: No Expert parameters found in model parameter list!")

    print("\n=== 🧱 BLOCO 3: Backward External (CPU) ===")

    # Grad same shape as output
    grad = nsos_ext.Tensor.ones(out.shape, nsos_ext.Device.CPU)

    try:
        model.backward_external(grad, ctx)
        print("Backward External executed successfully.")
    except Exception as e:
        print(f"BACKWARD FAILED: {e}")
        sys.exit(1)

    print("\n=== 🧱 BLOCO 4: Parameter Update (Safe) ===")

    params = model.parameters()
    print(f"Updating {len(params)} parameters...")

    lr = 0.01
    for i, p in enumerate(params):
        if p.grad.size == 0:
            continue # No grad?

        # Check copy_from / math safety
        # p.data = p.data - lr * p.grad
        # Safe implementation using copy_from

        # p.grad.mul(lr) -> Tensor
        # p.data.sub(...) -> Tensor
        # p.data.copy_from(...)

        update = p.grad.mul(lr)
        new_data = p.data.sub(update)
        p.data.copy_from(new_data)

    print("Parameter update complete.")

    print("\n=== ✅ SANITY CHECK PASSED ===")

if __name__ == "__main__":
    test_interop_cpu()
