import os
import sys
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def run_pass(seed):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    nsos_ext.set_seed(seed)

    # Model
    model = nsos_ext.JambaModel(1, 64, 100) # 1 layer
    ctx = nsos_ext.Context()

    # Input
    input_ids = [1, 2, 3, 4]

    # Forward
    hidden = model.forward_ids(input_ids, ctx)

    # To Numpy
    return np.array(hidden, copy=True)

def test_determinism():
    print("🔬 Running Determinism Golden Test...")

    # Run 1
    out1 = run_pass(42)

    # Run 2
    out2 = run_pass(42)

    # Run 3 (Diff Seed)
    out3 = run_pass(43)

    # Check 1 vs 2 (Should be Identical)
    if np.allclose(out1, out2, atol=1e-6):
        print("✅ Determinism Check (Same Seed): PASS")
    else:
        print("❌ Determinism Check (Same Seed): FAIL")
        diff = np.abs(out1 - out2).max()
        print(f"   Max Diff: {diff}")
        sys.exit(1)

    # Check 1 vs 3 (Should be Different)
    if not np.allclose(out1, out3, atol=1e-6):
        print("✅ Randomness Check (Diff Seed): PASS")
    else:
        print("❌ Randomness Check (Diff Seed): FAIL (Outputs identical)")
        sys.exit(1)

if __name__ == "__main__":
    test_determinism()
