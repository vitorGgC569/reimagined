import os
import sys
import torch

# Add build to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def test_cpu_safety():
    print("🔬 Testing CPU Safety...")
    t = nsos_ext.Tensor([2, 2], nsos_ext.Device.CPU)

    # Should work
    try:
        arr = t.numpy()
        print("✅ CPU .numpy() works.")
    except Exception as e:
        print(f"❌ CPU .numpy() failed: {e}")
        sys.exit(1)

    # Should work
    try:
        t_cpu = t.cpu()
        print("✅ CPU .cpu() works.")
    except Exception as e:
        print(f"❌ CPU .cpu() failed: {e}")
        sys.exit(1)

    # Should FAIL (Not GPU)
    try:
        iface = t.__cuda_array_interface__
        print("❌ CPU __cuda_array_interface__ should have failed but didn't.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"✅ CPU __cuda_array_interface__ correctly failed: {e}")
    except AttributeError:
        print("❌ Property not found?")

def test_gpu_mock():
    # We can't actually allocate GPU memory without a GPU.
    # But we can verify the binding exists.
    print("🔬 Testing GPU Binding Existence...")

    if not hasattr(nsos_ext.Tensor, "cpu"):
        print("❌ .cpu() method missing.")
        sys.exit(1)

    print("✅ Bindings present.")

if __name__ == "__main__":
    test_cpu_safety()
    test_gpu_mock()
