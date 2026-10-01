from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


WINDOWS_CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
WINDOWS_HIP_ROOTS = (
    Path(r"C:\TheRock\build"),
    Path(r"C:\Program Files\AMD\ROCm"),
)
_DLL_DIRECTORY_HANDLES: list[object] = []


def _version_key(path: Path) -> Tuple[int, ...]:
    match = re.search(r"v(\d+)(?:\.(\d+))?(?:\.(\d+))?", path.name.lower())
    if not match:
        return (0,)
    parts = [int(part) for part in match.groups() if part is not None]
    return tuple(parts) if parts else (0,)


def _dedupe_keep_order(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def find_cuda_roots(preferred_root: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []

    if preferred_root is not None:
        roots.append(Path(preferred_root))

    env_roots = []
    for env_name in ("NSOS_CUDA_ROOT", "CUDA_PATH"):
        value = os.environ.get(env_name)
        if value:
            env_roots.append(Path(value))

    versioned_envs = []
    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V") and value:
            versioned_envs.append(Path(value))
    versioned_envs.sort(key=_version_key, reverse=True)

    installed_roots = []
    if WINDOWS_CUDA_ROOT.exists():
        installed_roots = [path for path in WINDOWS_CUDA_ROOT.iterdir() if path.is_dir()]
        installed_roots.sort(key=_version_key, reverse=True)

    roots.extend(env_roots)
    roots.extend(versioned_envs)
    roots.extend(installed_roots)
    return [path for path in _dedupe_keep_order(roots) if path.exists()]


def find_cuda_bin_dirs(preferred_root: Optional[Path] = None) -> List[Path]:
    bins: List[Path] = []
    for root in find_cuda_roots(preferred_root):
        for suffix in ("bin\\x64", "bin"):
            candidate = root / suffix
            if candidate.exists():
                bins.append(candidate)
    return _dedupe_keep_order(bins)


def find_hip_roots(preferred_root: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []
    if preferred_root is not None:
        roots.append(Path(preferred_root))
    for env_name in ("NSOS_HIP_ROOT", "ROCM_PATH", "HIP_PATH"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    for installed_root in WINDOWS_HIP_ROOTS:
        if not installed_root.exists():
            continue
        if (installed_root / "bin").is_dir():
            roots.append(installed_root)
        roots.extend(
            child
            for child in installed_root.iterdir()
            if child.is_dir() and (child / "bin").is_dir()
        )
    return [path for path in _dedupe_keep_order(roots) if path.exists()]


def find_hip_bin_dirs(preferred_root: Optional[Path] = None) -> List[Path]:
    bins: List[Path] = []
    for root in find_hip_roots(preferred_root):
        for suffix in ("bin", "lib\\llvm\\bin"):
            candidate = root / suffix
            if candidate.is_dir():
                bins.append(candidate)
    return _dedupe_keep_order(bins)


def add_windows_runtime_dirs(
    build_dir: Path,
    preferred_cuda_root: Optional[Path] = None,
    preferred_hip_root: Optional[Path] = None,
) -> None:
    if os.name != "nt":
        return
    runtime_dirs = [Path(build_dir)]
    runtime_dirs.extend(find_cuda_bin_dirs(preferred_cuda_root))
    runtime_dirs.extend(find_hip_bin_dirs(preferred_hip_root))
    for runtime_dir in _dedupe_keep_order(runtime_dirs):
        if not runtime_dir.is_dir():
            continue
        handle = os.add_dll_directory(str(runtime_dir))
        _DLL_DIRECTORY_HANDLES.append(handle)


def detect_preferred_cuda_root() -> Optional[Path]:
    roots = find_cuda_roots()
    return roots[0] if roots else None


def parse_preferred_cuda_root(text: str | None) -> Optional[Path]:
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def parse_preferred_hip_root(text: str | None) -> Optional[Path]:
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def prepend_cuda_bin_to_path(preferred_root: Optional[Path] = None) -> None:
    if os.name != "nt":
        return
    bins = find_cuda_bin_dirs(preferred_root)
    if not bins:
        return
    existing = os.environ.get("PATH", "")
    prefix = os.pathsep.join(str(path) for path in bins)
    os.environ["PATH"] = prefix + (os.pathsep + existing if existing else "")


def describe_cuda_roots() -> List[str]:
    return [str(path) for path in find_cuda_roots()]


def describe_hip_roots() -> List[str]:
    return [str(path) for path in find_hip_roots()]


if __name__ == "__main__":
    for entry in describe_cuda_roots():
        sys.stdout.write(entry + "\n")
