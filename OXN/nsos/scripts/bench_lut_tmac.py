"""Microbench: LUT-TMAC kernel vs reference scalar kernel.

Runs `test_lut_tmac` style shapes through Python and times each path.
Reports tokens/s and absolute speedup so we can prove the win.

This is a microbench (pure GEMM, no model) so the numbers are upper-
bounds for a single linear layer.  The real-model speedup will be
diluted by attention, Mamba2 scan, embeddings — typically 30-50% of
the raw kernel speedup ends up at the model level.

Usage:
  python OXN/nsos/scripts/bench_lut_tmac.py --build-dir OXN/nsos/build-cuda-validation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def setup_nsos(build_dir: Path):
    if os.name == "nt":
        from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import nsos_ext  # type: ignore
    return nsos_ext


# Public shapes representative of a real Mamba/Jamba hybrid linear layer.
# (B, K, N) — batch, in_features, out_features.
DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (1, 128, 512),
    (1, 256, 1024),
    (8, 512, 2048),
    (16, 1024, 4096),
    (1, 2048, 8192),
]


def microbench(nsos, B: int, K: int, N: int, iters: int = 50, zero_density: float = 0.4):
    """Run the same GEMM N times via reference and via LUT-TMAC.

    Because we don't have a clean Python binding for the raw kernels,
    we exercise the WHOLE BitLinear.forward path with NSOS_TMAC_LUT_GEMM
    toggled.  Per-call overhead (quantization etc.) is identical
    between the two so the delta is the kernel's.
    """
    import numpy as np

    # Build a BitLinear layer
    layer = nsos.BitLinear(K, N) if hasattr(nsos, "BitLinear") else None
    if layer is None:
        raise RuntimeError(
            "BitLinear binding is required for the LUT-TMAC benchmark"
        )

    # Random input
    x = nsos.Tensor([B, K], nsos.Device.CPU, 0.0)
    arr = np.random.randn(B, K).astype("float32")
    # Copy via numpy if available
    try:
        np_view = x.numpy()
        np_view[:] = arr
    except Exception as exc:
        raise RuntimeError(
            "Benchmark input could not be copied into the NSOS tensor"
        ) from exc

    # ── Path A: default kernel (NSOS_TMAC_LUT_GEMM not set) ─────────
    os.environ.pop("NSOS_TMAC_LUT_GEMM", None)
    # Warmup
    for _ in range(3):
        _ = layer.forward(x)
    t0 = time.time()
    for _ in range(iters):
        y_default = layer.forward(x)
    t_default = (time.time() - t0) / iters

    # ── Path B: LUT-TMAC ──────────────────────────────────────────
    os.environ["NSOS_TMAC_LUT_GEMM"] = "1"
    for _ in range(3):
        _ = layer.forward(x)
    t0 = time.time()
    for _ in range(iters):
        y_lut = layer.forward(x)
    t_lut = (time.time() - t0) / iters
    os.environ.pop("NSOS_TMAC_LUT_GEMM", None)

    return {
        "B": B, "K": K, "N": N, "iters": iters,
        "default_ms": t_default * 1000.0,
        "lut_ms": t_lut * 1000.0,
        "speedup": t_default / max(t_lut, 1e-9),
        "default_tok_s": (B / t_default) if t_default > 0 else 0,
        "lut_tok_s": (B / t_lut) if t_lut > 0 else 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    nsos = setup_nsos(args.build_dir.resolve())
    print(f"=== LUT-TMAC microbench ===")
    print(f"  iters per shape: {args.iters}")
    print(f"  device: CPU (LUT path is CPU-only in MVP)")
    print()
    print(f"{'shape':<24} {'default ms':>12} {'lut ms':>10} {'speedup':>10}")
    print("-" * 60)
    any_data = False
    for B, K, N in DEFAULT_SHAPES:
        res = microbench(nsos, B, K, N, iters=args.iters)
        if res is None:
            print(f"  (no BitLinear python binding — skipped)")
            continue
        any_data = True
        print(f"  B={res['B']:<3} K={res['K']:<5} N={res['N']:<5} "
              f"{res['default_ms']:>12.3f} {res['lut_ms']:>10.3f} "
              f"{res['speedup']:>9.2f}x")
    if not any_data:
        print("(no shapes had bindings; bench skipped.  C++ test_lut_tmac "
              "remains the authoritative correctness check.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
