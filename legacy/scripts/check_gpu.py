import sys
import os

# Setup path
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import nsos_ext as nsos
    print(f"NSOS Loaded. GPU Support Check:")
    
    try:
        # Check if Device.GPU enum exists
        gpu_dev = nsos.Device.GPU
        print(f" - nsos.Device.GPU exists: {gpu_dev}")
        
        # Try to allocate a small tensor on GPU
        try:
            t = nsos.Tensor([10], nsos.Device.GPU)
            print(" - Allocation on GPU successful!")
            print(f" - Tensor device: {t.device}")
        except Exception as e:
            print(f" - Allocation on GPU failed: {e}")
            
    except AttributeError:
        print(" - nsos.Device.GPU does NOT exist.")
        
except ImportError as e:
    print(f"Failed to import nsos_ext: {e}")
