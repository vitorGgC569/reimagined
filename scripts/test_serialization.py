
import os
import sys
import numpy as np

# Path Setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found. Build extensions first.")
    sys.exit(1)

def test_serialization():
    print("=== 💾 Serialization & Parameter Access Test ===")

    # 1. Init Model
    print("Initializing Model...")
    model = nsos_ext.JambaModel(2, 64, 100) # Small model

    # 2. Modify Weights (to prove load works)
    print("Modifying weights...")
    params = model.parameters()
    if not params:
        print("Error: No parameters returned!")
        sys.exit(1)

    # Access first parameter
    p0 = params[0]
    # Check if we can get numpy view
    w0_np = np.array(p0.data, copy=False) # Zero-copy access?

    print(f"Param 0 Shape: {w0_np.shape}")
    print(f"Param 0 Mean (Before): {np.mean(w0_np):.4f}")

    # Modify in-place
    w0_np[:] += 1.0
    print(f"Param 0 Mean (Modified): {np.mean(w0_np):.4f}")

    # 3. Save
    save_path = "test_model.ox3"
    print(f"Saving to {save_path}...")
    model.save(save_path)

    if not os.path.exists(save_path):
        print("Error: File not created.")
        sys.exit(1)

    # 4. Load into NEW model
    print("Loading into new model...")
    model2 = nsos_ext.JambaModel(2, 64, 100)
    model2.load(save_path)

    # 5. Verify
    params2 = model2.parameters()
    w0_np_2 = np.array(params2[0].data, copy=False)

    print(f"Model 2 Param 0 Mean: {np.mean(w0_np_2):.4f}")

    diff = np.abs(np.mean(w0_np) - np.mean(w0_np_2))
    if diff < 1e-5:
        print("✅ Success: Weights match!")
    else:
        print(f"❌ Fail: Weight mismatch (Diff: {diff})")

    # Cleanup
    os.remove(save_path)

if __name__ == "__main__":
    test_serialization()
