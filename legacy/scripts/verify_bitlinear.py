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

def test_bitlinear():
    print("\n--- Testing New BitLinear Features ---")
    in_features = 64
    out_features = 128
    
    # 1. Initialization
    layer = nsos.BitLinear(in_features, out_features, True)
    print("✅ BitLinear initialized")
    
    # 2. Test repack_weights
    try:
        layer.repack_weights()
        print("✅ repack_weights() called successfully")
    except Exception as e:
        print(f"❌ repack_weights() failed: {e}")
        
    # 3. Test forward pass
    x = nsos.Tensor.random([1, 16, in_features])
    try:
        y = layer.forward(x)
        print(f"✅ Forward pass successful. Output shape: {y.shape}")
        
        # Check if output is non-zero
        norm = y.norm()
        print(f"✅ Output Norm: {norm:.4f}")
        if norm == 0:
            print("⚠️ Warning: Output is all zeros. Check quantization logic.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

    # 4. Test backward pass
    try:
        dy = nsos.Tensor.ones(y.shape)
        dx = layer.backward(dy)
        print(f"✅ Backward pass successful. Input grad shape: {dx.shape}")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

    print("\n--- Testing Control Methods ---")
    try:
        layer.set_use_hadamard(False)
        layer.set_mixed_precision(True)
        print("✅ set_use_hadamard and set_mixed_precision successful")
        
        y_bypass = layer.forward(x)
        print(f"✅ Forward (Bypass) successful. Norm: {y_bypass.norm():.4f}")
    except Exception as e:
        print(f"❌ Control methods failed: {e}")

if __name__ == "__main__":
    test_bitlinear()
