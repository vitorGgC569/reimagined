"""Microbench: bitmamba.cpp UNPACK_LUT + AVX2 SIMD path (Cherry-pick #1).

Compares three implementations of gemm_158bit:
  - scalar LUT  (existing default)        : NSOS_USE_LUT_SIMD unset
  - SIMD LUT    (new, this cherry-pick)   : NSOS_USE_LUT_SIMD=1
  - SIMD i8     (existing baseline)       : gemm_158bit_i8 direct

The three paths produce mathematically equivalent results (within FP +
int8-quantization tolerance for the SIMD path).  This script measures
THROUGHPUT, not correctness — correctness lives in
tests/test_bitmamba_lut.cpp.

Because the SIMD-vs-scalar dispatch in gemm_158bit_lut uses a process-
local lazy env cache (so we don't pay getenv() on every GEMM call), this
script spawns a fresh subprocess per measurement.  Each subprocess
selects exactly one path via env var, runs N iterations, and reports
tokens/s + latency.

Usage:
  python OXN/nsos/scripts/bench_bitmamba_lut.py --build-dir OXN/nsos/build-cuda-validation
  python OXN/nsos/scripts/bench_bitmamba_lut.py --shape 32 4096 4096 --iters 100

See OXN/nsos/docs/BITMAMBA_LUT_INTEGRATION.md for the integration design.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def setup_nsos(build_dir: Path):
    """Replica do helper in bench_lut_tmac.py — adds the .so to sys.path."""
    if os.name == "nt":
        try:
            from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
            add_windows_runtime_dirs(
                build_dir,
                parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
            )
        except ImportError:
            pass  # cuda_env optional on non-CUDA builds
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import nsos_ext  # type: ignore  # noqa: F401
    return nsos_ext


# Shapes representative of a Mamba/Jamba hybrid linear layer.
# (M, N, K) = (rows, out_cols, in_cols) — note K=in_cols matches the
# packed weight matrix's hidden dim.
DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (1,   640,  640),     # batch=1, d_model=640, projection
    (8,   640,  2560),    # FFN-style 1×4 expansion (M=batch, K=4*d, N=d)
    (32,  640,  640),     # batch=32, attention output projection
    (1,   4096, 4096),    # batch=1, larger model (single-user inference)
    (1,   16384, 640),    # batch=1, lm_head projection (vocab=16k)
]


def run_one_measurement(build_dir: Path, mode: str, M: int, N: int, K: int,
                         iters: int) -> Dict:
    """Run a single measurement in a fresh subprocess.

    mode = 'lut_scalar' | 'lut_simd' | 'i8'
    """
    script = f"""
import os, sys, time, json
sys.path.insert(0, {str(build_dir)!r})
sys.path.insert(0, {str(HERE)!r})
if os.name == 'nt':
    try:
        from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
        add_windows_runtime_dirs({str(build_dir)!r},
            parse_preferred_cuda_root(os.environ.get('NSOS_CUDA_ROOT')))
    except ImportError:
        pass
import nsos_ext

# This script intentionally minimal — relies on whatever testbed function
# the binding exposes for benching gemm_158bit_lut directly.  If the
# binding does not expose this (it currently doesn't), this measurement
# falls back to constructing a BitLinear, calling forward, and timing.

# Build a BitLinear of the right shape and time forward passes.
import numpy as np
np.random.seed(0)

bl = nsos_ext.BitLinear({K}, {N}, False) if hasattr(nsos_ext, 'BitLinear') else None
if bl is None:
    print(json.dumps({{'error': 'BitLinear not exposed via nsos_ext'}}))
    sys.exit(1)

# Random input tensor [M, K]
x_np = np.random.randn({M}, {K}).astype(np.float32)
x = nsos_ext.Tensor.from_numpy(x_np) if hasattr(nsos_ext.Tensor, 'from_numpy') else None
if x is None:
    print(json.dumps({{'error': 'Tensor.from_numpy not exposed'}}))
    sys.exit(1)

# Warmup
for _ in range(3):
    _ = bl.forward(x)

# Bench
t0 = time.perf_counter()
for _ in range(iters):
    _ = bl.forward(x)
t1 = time.perf_counter()

elapsed_total = t1 - t0
elapsed_per_iter = elapsed_total / {iters}
tokens_per_s = ({M}) / elapsed_per_iter if elapsed_per_iter > 0 else 0.0

print(json.dumps({{
    'mode': {mode!r},
    'M': {M}, 'N': {N}, 'K': {K},
    'iters': {iters},
    'elapsed_total_s': elapsed_total,
    'elapsed_per_iter_s': elapsed_per_iter,
    'tokens_per_s': tokens_per_s,
}}))
"""

    env = os.environ.copy()
    if mode == "lut_simd":
        env["NSOS_USE_LUT_SIMD"] = "1"
    elif mode == "lut_scalar":
        env["NSOS_USE_LUT_SIMD"] = "0"
    # 'i8' uses the default i8 path of BitLinear; LUT env irrelevant

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        return {
            "mode": mode, "M": M, "N": N, "K": K,
            "error": result.stderr.strip()[:500] or "non-zero exit",
        }
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {
            "mode": mode, "M": M, "N": N, "K": K,
            "error": f"could not parse output: {result.stdout[:200]}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, required=True,
                    help="Directory containing nsos_ext.*.so / .pyd")
    ap.add_argument("--iters", type=int, default=20,
                    help="Iterations per measurement (default 20)")
    ap.add_argument("--shape", type=int, nargs=3, action="append",
                    metavar=("M", "N", "K"),
                    help="Override default shapes; can pass multiple --shape M N K")
    ap.add_argument("--report-path", type=Path,
                    help="Write JSON report to this path (in addition to stdout)")
    args = ap.parse_args()

    shapes = args.shape if args.shape else DEFAULT_SHAPES

    print(f"Bitmamba LUT bench — Cherry-pick #1")
    print(f"build-dir: {args.build_dir}")
    print(f"shapes:    {shapes}")
    print(f"iters:     {args.iters}")
    print()

    rows = []
    for M, N, K in shapes:
        for mode in ("lut_scalar", "lut_simd", "i8"):
            print(f"[run] mode={mode:<11s} M={M:<5d} N={N:<6d} K={K:<6d} ... ",
                  end="", flush=True)
            r = run_one_measurement(args.build_dir, mode, M, N, K, args.iters)
            rows.append(r)
            if "error" in r:
                print(f"ERR: {r['error']}")
            else:
                print(f"{r['tokens_per_s']:.1f} tok/s "
                      f"({r['elapsed_per_iter_s']*1000:.2f}ms/iter)")

    # Summary table
    print()
    print("=" * 70)
    print(f"{'shape':<22s} {'scalar':>10s} {'simd':>10s} {'i8':>10s}  speedup")
    print("-" * 70)
    for M, N, K in shapes:
        shape_str = f"({M:>3d}, {N:>5d}, {K:>5d})"
        s = next((r for r in rows if r.get('M')==M and r.get('N')==N
                  and r.get('K')==K and r.get('mode')=='lut_scalar'
                  and 'error' not in r), None)
        i = next((r for r in rows if r.get('M')==M and r.get('N')==N
                  and r.get('K')==K and r.get('mode')=='lut_simd'
                  and 'error' not in r), None)
        b = next((r for r in rows if r.get('M')==M and r.get('N')==N
                  and r.get('K')==K and r.get('mode')=='i8'
                  and 'error' not in r), None)
        s_tps = s['tokens_per_s'] if s else 0.0
        i_tps = i['tokens_per_s'] if i else 0.0
        b_tps = b['tokens_per_s'] if b else 0.0
        speedup = i_tps / s_tps if s_tps > 0 else 0.0
        print(f"{shape_str:<22s} {s_tps:>10.1f} {i_tps:>10.1f} {b_tps:>10.1f}  "
              f"{speedup:>5.2f}x vs scalar")

    report = {
        "build_dir": str(args.build_dir),
        "shapes": shapes,
        "iters": args.iters,
        "rows": rows,
    }
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
