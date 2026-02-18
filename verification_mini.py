
import sys
import os

build_dir = os.path.join(os.getcwd(), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import nsos_ext as nsos
    print("✅ nsos_ext imported successfully!")
    t = nsos.Tensor.zeros([2, 2], nsos.Device.CPU)
    print(f"✅ Tensor creation works: {t.shape}")
    
    # Check JambaModel and MoE routing
    model = nsos.JambaModel(2, 64, 128)
    print("✅ JambaModel created.")
    
    # Forward pass
    ctx = nsos.Context()
    x = nsos.Tensor.random([1, 8, 64], nsos.Device.CPU)
    h = model.forward(x, ctx)
    print(f"✅ Forward pass works: {h.shape}")
    
    # Backward pass
    dy = nsos.Tensor.random(h.shape, nsos.Device.CPU)
    model.backward_external(dy, ctx)
    print("✅ Backward pass works (Exercises MoE Scatter/Gather if applicable)")
    
    print("\nALL BASIC VERIFICATIONS PASSED!")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
