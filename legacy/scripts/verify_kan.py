import os
import sys

# Add build path to sys.path with priority
build_path = r"c:\Users\VitorGGc\Desktop\Pantheon-Oxtav1-15338770387506525964\OXN\build\Release"
if build_path not in sys.path:
    sys.path.insert(0, build_path)

# Add CUDA bin to DLL path for Windows
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if os.path.exists(cuda_path) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_path)

try:
    import nsos_ext as nsos
    print(f"✅ nsos_ext loaded from: {nsos.__file__}")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

def test_kan_optimized():
    print("\n--- Testing Optimized BitFastKANLayer ---")
    in_features = 16
    out_features = 32
    grid_size = 5
    
    # 1. Initialization
    layer = nsos.BitFastKANLayer(in_features, out_features, grid_size)
    print("✅ BitFastKANLayer initialized")
    
    # Check parameters
    params = layer.parameters()
    print(f"✅ Parameters found: {[p.name for p in params]}")
    expected_names = ["kan.base", "kan.rbf", "kan.centers", "kan.sigma"]
    for name in expected_names:
        if not any(name in p.name for p in params):
             print(f"⚠️ Warning: Missing parameter {name}")

    # 2. Forward Pass
    x = nsos.Tensor.random([1, 8, in_features])
    try:
        y = layer.forward(x)
        print(f"✅ Forward pass successful. Output shape: {y.shape}, Norm: {y.norm():.4f}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 3. Backward Pass
    try:
        dy = nsos.Tensor.ones(y.shape)
        dx = layer.backward(dy)
        print(f"✅ Backward pass successful. Input grad shape: {dx.shape}")
        
        # Check if centers and sigma got gradients
        for p in params:
            grad_norm = p.grad.norm() if p.grad else 0.0
            print(f"   - Param '{p.name}' Grad Norm: {grad_norm:.6f}")
            if grad_norm == 0 and "centers" not in p.name: # Centers might be 0 if input exactly at centers, but sigma/weights shouldn't be
                 print(f"⚠️ Warning: Gradient for {p.name} is zero!")
                 
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

    # 4. Numerics check (ensure no NaNs)
    import numpy as np
    y_np = np.array(y)
    if np.isnan(y_np).any():
        print("❌ ERROR: NaNs detected in output!")
    else:
        print("✅ No NaNs detected in output")

if __name__ == "__main__":
    test_kan_optimized()
