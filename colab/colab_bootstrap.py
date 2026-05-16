"""NSOS v11 — Colab bootstrap.

This script is called from train_v11.ipynb to:
    1. Detect the Colab GPU (T4 / A100 / L4 / V100) → pick the matching
       train_curriculum.py profile name.
    2. Check Google Drive for a cached prebuilt nsos_ext .so for the
       current git commit; if found, copy it instead of rebuilding.
    3. Otherwise: apt-install build deps, run cmake + ninja, build the
       .so (with NSOS_CUDA_ARCHITECTURES detected for the runtime GPU),
       and upload the result to Drive for future sessions.
    4. Check Drive for the cached distillation_bundle_v11; if found,
       extract; otherwise return a flag so the caller can fetch it.

Why a separate script (not just notebook cells):
    - Notebook cells get re-executed often by users by accident; this
      keeps the heavy logic idempotent and re-runnable.
    - Easier to unit-test the bootstrap pieces in isolation.
    - Same script works from VS Code Colab extension or terminal Colab.

Environment expected by this script:
    - Running on Linux x86_64 (Colab's Ubuntu).
    - `nvidia-smi`, `nvcc`, and `apt` available.
    - /content/drive/MyDrive/ mounted (caller does drive.mount() first).
    - $REPO_ROOT env var pointing to the cloned repo (default
      /content/reimagined-main).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Paths ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/content/reimagined-main"))
DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/nsos_v11"))
BOOTSTRAP_CACHE = DRIVE_ROOT / "_bootstrap"
DATASET_CACHE = DRIVE_ROOT / "_datasets_processed"
RAW_DATASET_CACHE = DRIVE_ROOT / "_datasets_raw"
RUNS_ROOT = DRIVE_ROOT / "runs"
NSOS_BUILD_DIR = REPO_ROOT / "OXN/nsos/build-colab"
NSOS_EXT_SO_NAME = "nsos_ext.cpython-311-x86_64-linux-gnu.so"

# Map: GPU name keyword → (profile_40m, profile_80m)
# Default for v11 is the 80M variant — user opted for "Option 2" sweet
# spot (Chinchilla-optimal at 1.6B token budget).  Override via the
# NSOS_PARAM_SCALE env var ("40m" to fall back to the smaller arch).
GPU_PROFILE_MAP: Dict[str, Dict[str, str]] = {
    "T4":   {"40m": "hybrid_v11_colab_t4",   "80m": "hybrid_v11_colab_t4_80m"},
    "A100": {"40m": "hybrid_v11_colab_a100", "80m": "hybrid_v11_colab_a100_80m"},
    "L4":   {"40m": "hybrid_v11_colab_l4",   "80m": "hybrid_v11_colab_l4_80m"},
    "V100": {"40m": "hybrid_v11_colab_t4",   "80m": "hybrid_v11_colab_t4_80m"},
    "H100": {"40m": "hybrid_v11_colab_a100", "80m": "hybrid_v11_colab_a100_80m"},
}

# Default param scale ("40m" or "80m").  Override via NSOS_PARAM_SCALE.
# Default is "40m" because at the v11 token budget (~42K samples, ~21M
# tokens) Chinchilla-optimal sits closer to 40M params than 80M, AND
# the 40M variant runs ~2x faster per step (fewer FLOPs) so the user
# completes the curriculum in fewer Colab sessions.
DEFAULT_PARAM_SCALE = os.environ.get("NSOS_PARAM_SCALE", "40m").lower()

# Map: GPU keyword → CUDA compute capability (for NSOS_CUDA_ARCHITECTURES)
GPU_ARCH_MAP: Dict[str, str] = {
    "T4":   "75",
    "V100": "70",
    "L4":   "89",
    "A100": "80",
    "H100": "90",
}


# ── Helpers ─────────────────────────────────────────────────────────────
def _run(cmd: list, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"[bootstrap] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kwargs)


def detect_gpu() -> Tuple[str, str, str]:
    """Returns (gpu_friendly_name, profile_name, cuda_arch).

    Raises RuntimeError if no supported GPU is detected.  The caller
    can catch this and decide to run on CPU (not supported for training)
    or to error out cleanly with a message.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise RuntimeError(f"nvidia-smi failed — no NVIDIA GPU? {e}")

    gpu_name = result.stdout.strip()
    if not gpu_name:
        raise RuntimeError("nvidia-smi returned empty — no GPU allocated.")

    scale = DEFAULT_PARAM_SCALE
    if scale not in ("40m", "80m"):
        print(f"[bootstrap] WARNING: NSOS_PARAM_SCALE={scale!r} unknown, "
              f"falling back to 80m")
        scale = "80m"

    # Match against known GPU archs.
    for keyword, profile_map in GPU_PROFILE_MAP.items():
        if keyword in gpu_name:
            profile = profile_map[scale]
            arch = GPU_ARCH_MAP[keyword]
            print(f"[bootstrap] detected GPU: {gpu_name!r} -> "
                  f"profile={profile}  (scale={scale}, sm_{arch})", flush=True)
            return gpu_name, profile, arch

    raise RuntimeError(
        f"Unsupported GPU: {gpu_name!r}. "
        f"Known: {list(GPU_PROFILE_MAP.keys())}.  Add a profile in "
        f"train_curriculum.py and update GPU_PROFILE_MAP here."
    )


def current_git_sha() -> str:
    """Short git SHA of the cloned repo, used to key the binary cache."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def cached_so_path(sha: str, arch: str) -> Path:
    """Where the prebuilt .so for this (commit, arch) tuple lives in Drive."""
    return BOOTSTRAP_CACHE / f"{NSOS_EXT_SO_NAME}.{sha}.sm{arch}"


def install_build_deps() -> None:
    """apt-install cmake/ninja/pybind11 once per session.  Cheap if already there."""
    _run(["apt-get", "update", "-qq"], check=False)
    _run(["apt-get", "install", "-y", "-qq",
          "cmake", "ninja-build", "build-essential"], check=True)
    _run([sys.executable, "-m", "pip", "install", "-q", "pybind11>=2.10", "numpy"], check=True)


def build_nsos_ext(arch: str) -> Path:
    """Build the Linux .so for this Colab session.  Returns path to the .so.

    Skips rebuild if Drive has a cache for (current_sha, arch).  Always
    runs in a clean build dir to avoid stale CMake state from a previous
    run that may have used a different arch.
    """
    sha = current_git_sha()
    cached = cached_so_path(sha, arch)
    target = NSOS_BUILD_DIR / NSOS_EXT_SO_NAME

    BOOTSTRAP_CACHE.mkdir(parents=True, exist_ok=True)
    NSOS_BUILD_DIR.parent.mkdir(parents=True, exist_ok=True)

    # ── Fast path: cache hit ─────────────────────────────────────────────
    if cached.exists():
        print(f"[bootstrap] cache HIT: {cached.name}  (skipping build)")
        NSOS_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        return target

    # ── Slow path: full build ────────────────────────────────────────────
    print(f"[bootstrap] cache MISS for sha={sha} arch=sm_{arch} — building")
    install_build_deps()

    if NSOS_BUILD_DIR.exists():
        shutil.rmtree(NSOS_BUILD_DIR)
    NSOS_BUILD_DIR.mkdir(parents=True)

    nsos_src = REPO_ROOT / "OXN/nsos"
    _run([
        "cmake", "-G", "Ninja",
        "-S", str(nsos_src),
        "-B", str(NSOS_BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DNSOS_ENABLE_CUDA=ON",
        "-DNSOS_BUILD_PYTHON=ON",
        "-DNSOS_BUILD_TESTS=OFF",
        "-DNSOS_BUILD_CLI=OFF",
        "-DNSOS_BUILD_API=OFF",
        "-DNSOS_BUILD_OXTAMEM=OFF",
        f"-DNSOS_CUDA_ARCHITECTURES={arch}",
    ], check=True)

    _run(["cmake", "--build", str(NSOS_BUILD_DIR), "--config", "Release",
          f"--parallel", str(os.cpu_count() or 4)], check=True)

    if not target.exists():
        # Linux build produces a slightly different name depending on
        # Python ABI tag; find it.
        candidates = list(NSOS_BUILD_DIR.glob("nsos_ext*.so"))
        if not candidates:
            raise RuntimeError(
                f"Build succeeded but no nsos_ext.so found in {NSOS_BUILD_DIR}"
            )
        target = candidates[0]
        print(f"[bootstrap] resolved .so name -> {target.name}")

    # ── Cache the result back to Drive ───────────────────────────────────
    shutil.copy2(target, cached)
    print(f"[bootstrap] cached -> {cached}")
    return target


def has_processed_bundle() -> bool:
    """Check if distillation_bundle_v11 is already in Drive cache."""
    return (DATASET_CACHE / "distillation_bundle_v11.zip").exists()


def fetch_processed_bundle_to_repo() -> bool:
    """Copy the cached bundle from Drive into the repo's scripts dir.

    Returns True if extracted, False if no cache (caller must build it).
    """
    cached_zip = DATASET_CACHE / "distillation_bundle_v11.zip"
    if not cached_zip.exists():
        return False

    target_dir = REPO_ROOT / "OXN/nsos/scripts/distillation_bundle_v11"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    _run(["unzip", "-q", str(cached_zip), "-d", str(target_dir)], check=True)
    print(f"[bootstrap] extracted bundle -> {target_dir}")
    return True


def list_drive_runs() -> list:
    """Return list of existing run dirs in Drive (latest first)."""
    if not RUNS_ROOT.exists():
        return []
    return sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()],
                  key=lambda p: p.name, reverse=True)


def setup_environment(verbose: bool = True) -> Dict[str, str]:
    """High-level entry point called from notebook cell 1.

    Returns a dict with everything the rest of the notebook needs:
        gpu_name, profile, arch, build_dir, ext_so, bundle_ready,
        drive_runs, sha.
    """
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_CACHE.mkdir(parents=True, exist_ok=True)
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    gpu_name, profile, arch = detect_gpu()
    sha = current_git_sha()
    ext_so = build_nsos_ext(arch)
    bundle_ready = fetch_processed_bundle_to_repo()
    drive_runs = list_drive_runs()

    info = {
        "gpu_name": gpu_name,
        "profile": profile,
        "arch": arch,
        "sha": sha,
        "build_dir": str(NSOS_BUILD_DIR),
        "ext_so": str(ext_so),
        "bundle_ready": bundle_ready,
        "drive_runs": [str(p.name) for p in drive_runs],
    }
    if verbose:
        print("\n[bootstrap] ============================================")
        print("[bootstrap] environment ready")
        for k, v in info.items():
            print(f"[bootstrap]   {k:<14} = {v}")
        print("[bootstrap] ============================================\n")
    return info


if __name__ == "__main__":
    info = setup_environment()
    sys.stdout.write(json.dumps(info, indent=2) + "\n")
