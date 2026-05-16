"""Detect installed NVIDIA GPU compute capabilities and emit a CMake
arch list suitable for -DNSOS_CUDA_ARCHITECTURES.

Usage:
    python gpu_arch_detect.py                # prints semicolon list, e.g. "75"
    python gpu_arch_detect.py --explain      # human-readable summary
    python gpu_arch_detect.py --fallback all # if no GPU: print "61;75;80;86"

Detection order:
    1. NSOS_CUDA_ARCHITECTURES env var (explicit override)
    2. `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
       (works in both Windows and Linux/Colab)
    3. `nvcc --list-gpu-arch` (fallback if smi missing but toolkit present)
    4. --fallback value (default: "61;75;80;86" — all common archs)

Compute capability map (so we can sanity-check):
    50  Maxwell        — Tesla M40 (deprecated)
    52  Maxwell v2     — GTX 980, M40
    53  Maxwell mobile — Jetson Nano
    60  Pascal         — Tesla P100
    61  Pascal         — GTX 1050 Ti / 1060 / 1070 / 1080 / Titan Xp
    70  Volta          — Tesla V100, Titan V
    72  Volta mobile   — Jetson AGX Xavier
    75  Turing         — RTX 2060/2070/2080, T4, Quadro RTX
    80  Ampere         — A100 (data center, 40/80 GB)
    86  Ampere         — RTX 3060/3070/3080/3090, A10, A40, A6000
    87  Ampere mobile  — Jetson Orin
    89  Ada Lovelace   — RTX 4060/4070/4080/4090, L40
    90  Hopper         — H100, H200
   100  Blackwell      — B100, B200
   120  Blackwell      — RTX 5080/5090

The script never invents arches we cannot validate via the toolkit;
it only echoes what was detected (or the explicit fallback).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional


# Common arches that the NSOS kernels are known to build cleanly for.
# Anything detected outside this set is reported as-is but flagged in --explain.
KNOWN_NSOS_ARCHS = {50, 52, 60, 61, 70, 72, 75, 80, 86, 87, 89, 90}

# Default fallback when no GPU detected and no override given.
# Pascal+Turing+Ampere covers GTX 10xx / T4 / RTX 30xx / A100.
DEFAULT_FALLBACK = "61;75;80;86"

# Friendly name lookup for --explain.
ARCH_NAMES = {
    50: "Maxwell",
    52: "Maxwell v2",
    53: "Maxwell mobile",
    60: "Pascal",
    61: "Pascal (consumer)",
    70: "Volta",
    72: "Volta mobile",
    75: "Turing (T4 / RTX 20xx)",
    80: "Ampere (A100)",
    86: "Ampere (RTX 30xx, A10, A40)",
    87: "Ampere mobile",
    89: "Ada Lovelace (RTX 40xx)",
    90: "Hopper (H100)",
    100: "Blackwell DC",
    120: "Blackwell consumer",
}


def _parse_compute_cap(raw: str) -> Optional[int]:
    """Convert '7.5' or '8.0' into 75 or 80 (the int CMake wants)."""
    text = raw.strip()
    if not text:
        return None
    match = re.match(r"^(\d+)\.(\d+)$", text)
    if match:
        return int(match.group(1)) * 10 + int(match.group(2))
    if text.isdigit() and len(text) in (2, 3):
        return int(text)
    return None


def detect_via_nvidia_smi() -> List[int]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        result = subprocess.run(
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    arches: List[int] = []
    seen = set()
    for line in result.stdout.splitlines():
        cap = _parse_compute_cap(line)
        if cap is not None and cap not in seen:
            seen.add(cap)
            arches.append(cap)
    return sorted(arches)


def detect_via_nvcc_list() -> List[int]:
    """Fall back to `nvcc --list-gpu-arch` when nvidia-smi is missing
    but the toolkit is installed.  This won't tell us the *host* GPU,
    but it returns the set of archs the *installed nvcc* can target —
    a safe superset for a build-time hint."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return []
    try:
        result = subprocess.run(
            [nvcc, "--list-gpu-arch"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    arches: List[int] = []
    seen = set()
    for line in result.stdout.splitlines():
        match = re.search(r"compute_(\d+)", line)
        if match:
            cap = int(match.group(1))
            if cap not in seen:
                seen.add(cap)
                arches.append(cap)
    return sorted(arches)


def detect_arches(fallback: str = DEFAULT_FALLBACK) -> List[int]:
    """Returns the resolved arch list, preferring host GPU over toolkit."""
    override = os.environ.get("NSOS_CUDA_ARCHITECTURES")
    if override:
        resolved: List[int] = []
        for chunk in override.replace(",", ";").split(";"):
            cap = _parse_compute_cap(chunk)
            if cap is not None:
                resolved.append(cap)
        if resolved:
            return sorted(set(resolved))

    detected = detect_via_nvidia_smi()
    if detected:
        return detected

    detected = detect_via_nvcc_list()
    if detected:
        # nvcc lists EVERY arch it supports — clamp to the modern subset
        # we ship kernels for so we don't waste compile time.
        clamped = [a for a in detected if a in KNOWN_NSOS_ARCHS]
        if clamped:
            return clamped
        return detected

    resolved = []
    for chunk in fallback.replace(",", ";").split(";"):
        cap = _parse_compute_cap(chunk)
        if cap is not None:
            resolved.append(cap)
    return sorted(set(resolved)) if resolved else [61, 75, 80, 86]


def format_for_cmake(arches: List[int]) -> str:
    return ";".join(str(a) for a in arches)


def describe(arches: List[int]) -> str:
    lines = []
    for arch in arches:
        name = ARCH_NAMES.get(arch, "Unknown")
        flag = "" if arch in KNOWN_NSOS_ARCHS else "  [UNTESTED]"
        lines.append(f"  sm_{arch}  ({name}){flag}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fallback",
        default=DEFAULT_FALLBACK,
        help=(
            "Arch list to use when no GPU is detected. "
            "Use 'all' for 61;75;80;86 (PSU-friendly multi-arch build), "
            "or 'native' to try the local GPU only."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print a human-readable summary instead of the bare list.",
    )
    args = parser.parse_args()

    fallback = args.fallback
    if fallback == "all":
        fallback = DEFAULT_FALLBACK

    arches = detect_arches(fallback=fallback)

    if args.explain:
        source = "env override"
        if not os.environ.get("NSOS_CUDA_ARCHITECTURES"):
            if detect_via_nvidia_smi():
                source = "nvidia-smi (host GPU)"
            elif detect_via_nvcc_list():
                source = "nvcc --list-gpu-arch (toolkit)"
            else:
                source = f"fallback ({fallback})"
        sys.stdout.write(f"NSOS CUDA architectures (source: {source}):\n")
        sys.stdout.write(describe(arches) + "\n")
        sys.stdout.write(f"\nCMake flag: -DNSOS_CUDA_ARCHITECTURES=\"{format_for_cmake(arches)}\"\n")
    else:
        sys.stdout.write(format_for_cmake(arches) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
