# pyinstaller_spec.spec — explicit PyInstaller build configuration for the
# OxtaTrainer standalone bundle.
#
# Why a spec file instead of CLI flags: lets us bundle the pre-compiled
# nsos_ext.pyd reliably across versions of PyInstaller, attach CUDA runtime
# DLLs explicitly, and strip everything we don't need (Tk, IDLE, etc.) to
# keep the bundle lean.
#
# Build:  pyinstaller pyinstaller_spec.spec --noconfirm
# Output: dist/OxtaTrainer/    (--onedir mode, one folder with .exe + DLLs)
#
# Run as: pyinstaller is invoked from build_standalone.bat / .sh which
# sets the NSOS_EXT_PYD and CUDA_DIR environment variables this spec reads.

import os
import sys
from pathlib import Path

# ── Paths injected by build script via env vars ────────────────────────────
NSOS_EXT_PYD = os.environ.get("NSOS_EXT_PYD")
if not NSOS_EXT_PYD or not Path(NSOS_EXT_PYD).exists():
    raise SystemExit(
        f"NSOS_EXT_PYD not set or doesn't exist: {NSOS_EXT_PYD!r}.\n"
        f"  Set NSOS_EXT_PYD to the absolute path of the compiled .pyd before running pyinstaller."
    )

CUDA_DIR = os.environ.get("CUDA_DIR") or os.environ.get("CUDA_PATH")
if not CUDA_DIR:
    raise SystemExit(
        "CUDA_DIR or CUDA_PATH not set.\n"
        "  Set CUDA_DIR to your CUDA Toolkit root (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5)."
    )

CUDA_BIN = Path(CUDA_DIR) / "bin"

# ── CUDA runtime DLLs to bundle ────────────────────────────────────────────
# nsos_ext.pyd depends on a small set of CUDA libraries.  We bundle them
# explicitly so the friend doesn't need to ensure his PATH is set up
# correctly — the .exe is self-contained.
CUDA_DLL_PATTERNS = [
    "cudart64_*.dll",     # CUDA runtime
    "cublas64_*.dll",     # cuBLAS
    "cublasLt64_*.dll",   # cuBLAS LT (BF16 path)
    "curand64_*.dll",     # cuRAND (RNG)
    "cusparse64_*.dll",   # cuSPARSE (some kernels)
]

cuda_binaries = []
for pattern in CUDA_DLL_PATTERNS:
    for dll in CUDA_BIN.glob(pattern):
        cuda_binaries.append((str(dll), "."))  # destination = root of bundle

if not cuda_binaries:
    print(f"WARNING: no CUDA DLLs matched {CUDA_DLL_PATTERNS} in {CUDA_BIN}")

# ── Native engine module ───────────────────────────────────────────────────
# We add it as a binary so PyInstaller doesn't try to analyze its imports
# (it can't — it's a compiled C++ extension).
engine_binary = (NSOS_EXT_PYD, ".")

# ── Excluded modules (trim bundle size) ────────────────────────────────────
EXCLUDES = [
    "tkinter", "_tkinter",
    "IDLE", "lib2to3",
    "test", "unittest",
    "matplotlib", "scipy",
    "pandas",  # not used by the trainer
    "jupyter", "IPython",
]

# ── Analysis ───────────────────────────────────────────────────────────────
SCRIPT = "trainer_main.py"

a = Analysis(
    [SCRIPT],
    pathex=[str(Path.cwd())],
    binaries=cuda_binaries + [engine_binary],
    datas=[],  # data/ is shipped as a sibling folder, NOT bundled into .exe
    hiddenimports=[
        "nsos_ext",
        "zstandard",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir mode
    name="OxtaTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX often triggers AV false positives + slows CUDA init
    console=True,            # need stdout for the operator to see progress
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OxtaTrainer",       # produces dist/OxtaTrainer/
)
