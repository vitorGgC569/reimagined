"""PyInstaller runtime hook — registers the frozen bundle directory as a
DLL search path so nsos_ext.pyd can find the bundled cudart64_*.dll /
cublas64_*.dll without depending on the user's PATH or CUDA Toolkit.

Loaded automatically by PyInstaller before any user code runs (see
pyinstaller_spec.spec :: runtime_hooks).
"""
import os
import sys
from pathlib import Path

if hasattr(os, "add_dll_directory") and getattr(sys, "frozen", False):
    _exe_dir = Path(sys.executable).parent
    for _candidate in (_exe_dir, _exe_dir / "_internal"):
        if _candidate.is_dir():
            try:
                os.add_dll_directory(str(_candidate))
            except (OSError, FileNotFoundError):
                pass
