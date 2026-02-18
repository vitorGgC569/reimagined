import ctypes
import os

def check(name, path=None):
    print(f"Checking {name}...", end=" ")
    try:
        if path:
            # For Python 3.8+, we often need to add the directory
            os.add_dll_directory(os.path.dirname(path))
            ctypes.WinDLL(path)
        else:
            ctypes.WinDLL(name)
        print("OK")
    except Exception as e:
        print(f"MISSING or ERROR: {e}")

# Try to add CUDA path globally for this process
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if os.path.exists(cuda_bin_path):
    print(f"Adding DLL directory: {cuda_bin_path}")
    os.add_dll_directory(cuda_bin_path)

check("vcomp140.dll")
check("cudart64_12.dll")

try:
    import nsos_ext
    print("SUCCESS: nsos_ext imported!")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
