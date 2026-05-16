import os
import sys
import ctypes

# Path to the .pyd file
pyd_path = r"C:\Users\VitorGGc\Desktop\Pantheon-Oxtav1-15338770387506525964\OXN\build\Release\nsos_ext.cp311-win_amd64.pyd"

print(f"Checking {pyd_path}...")
if not os.path.exists(pyd_path):
    print("Error: File not found!")
    sys.exit(1)

# Try adding common locations to DLL path (Python 3.8+)
if hasattr(os, "add_dll_directory"):
    # CUDA
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
    if os.path.exists(cuda_bin):
        print(f"Adding DLL directory: {cuda_bin}")
        os.add_dll_directory(cuda_bin)
    
    # Also add the directory of the .pyd itself
    pyd_dir = os.path.dirname(pyd_path)
    print(f"Adding DLL directory: {pyd_dir}")
    os.add_dll_directory(pyd_dir)

try:
    print("Attempting to load with ctypes.WinDLL...")
    # Loading as a data file or similar to see dependencies
    handle = ctypes.WinDLL(pyd_path)
    print("Success loading with ctypes.WinDLL!")
except Exception as e:
    print(f"Ctypes load failed: {e}")
    # On Windows, this often gives a more descriptive error if we look at the Win32 error code
    import traceback
    traceback.print_exc()

try:
    print("\nAttempting standard import...")
    sys.path.insert(0, os.path.dirname(pyd_path))
    import nsos_ext
    print("Standard import success!")
except ImportError as e:
    print(f"Standard import failed: {e}")
