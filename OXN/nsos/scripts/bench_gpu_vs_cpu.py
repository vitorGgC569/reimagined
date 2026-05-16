"""bench_gpu_vs_cpu.py
==========================

Direct CPU vs GPU performance benchmark for the operations the GPU
optimization plan touches:

  * Tensor matmul (baseline reduction)
  * BitLinear forward (default float matmul path)
  * BitLinear forward with set_gpu_packed_inference(True) (Phase 5b __dp4a)
  * Mamba2SSD forward, full-sequence (Phase 3 selective_scan dispatch)
  * MoERouter forward (Phase 4 fast path)

Outputs a JSON report with per-op CPU/GPU mean latencies + speedup.
Honest: no warmup is hidden, no fake numbers; if GPU is slower than
CPU on this hardware (small batches on GTX 1050 Ti is plausible due
to launch overhead), the report will say so.

Run:
  $env:PYTHONIOENCODING = "utf-8"
  python bench_gpu_vs_cpu.py --build-dir <path-to-build-cuda-validation>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


def _load_nsos(build_dir: Path):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        # Add the build dir AND the CUDA toolkit bin dirs to the DLL
        # search list so cudart64_*.dll resolves at module-load time.
        candidates = [
            build_dir,
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"),
        ]
        for path in candidates:
            try:
                if path.exists():
                    os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
    import nsos_ext  # type: ignore
    return nsos_ext


def time_call(label: str, fn: Callable[[], Any], *, repeat: int = 20,
              warmup: int = 5) -> Dict[str, Any]:
    """Warm + measure.  Returns a dict with min/mean/median latencies."""
    for _ in range(max(warmup, 0)):
        fn()
    samples: List[float] = []
    for _ in range(max(repeat, 1)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "label": label,
        "repeat": len(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "mean_s": statistics.fmean(samples),
        "median_s": statistics.median(samples),
    }


def speedup(cpu_mean: float, gpu_mean: float) -> float:
    if gpu_mean <= 0:
        return float("nan")
    return cpu_mean / gpu_mean


# ────────────────────────────────────────────────────────────────────
# Individual benches
# ────────────────────────────────────────────────────────────────────

def bench_matmul(nsos, M: int, K: int, N: int, *, repeat: int) -> Dict[str, Any]:
    a_cpu = nsos.Tensor.random([M, K], nsos.Device.CPU)
    b_cpu = nsos.Tensor.random([K, N], nsos.Device.CPU)
    a_gpu = a_cpu.to(nsos.Device.GPU)
    b_gpu = b_cpu.to(nsos.Device.GPU)

    cpu = time_call(
        f"matmul[{M}x{K}x{N}] cpu",
        lambda: a_cpu.matmul(b_cpu),
        repeat=repeat,
    )
    # Force GPU sync by materializing back to CPU each iteration so the
    # measured time includes kernel runtime + sync, not just dispatch.
    gpu = time_call(
        f"matmul[{M}x{K}x{N}] gpu",
        lambda: a_gpu.matmul(b_gpu).cpu(),
        repeat=repeat,
    )
    return {
        "op": "matmul",
        "shape": {"M": M, "K": K, "N": N},
        "cpu": cpu,
        "gpu": gpu,
        "speedup_cpu_div_gpu": speedup(cpu["mean_s"], gpu["mean_s"]),
    }


def bench_bitlinear(nsos, in_features: int, out_features: int, batch: int, *,
                    repeat: int, dp4a: bool) -> Dict[str, Any]:
    """BitLinear forward CPU (packed path) vs GPU (float OR __dp4a)."""
    cpu_layer = nsos.BitLinear(in_features, out_features, True)
    gpu_layer = nsos.BitLinear(in_features, out_features, True)

    # Mirror weights (random init differs across instances otherwise)
    cpu_params = cpu_layer.parameters()
    gpu_params = gpu_layer.parameters()
    for cp, gp in zip(cpu_params, gpu_params):
        gp.data.copy_from(cp.data)

    gpu_layer.to(nsos.Device.GPU)
    cpu_layer.set_precision_mode(2)
    gpu_layer.set_precision_mode(2)
    cpu_layer.set_reference_path(False)
    gpu_layer.set_reference_path(False)
    cpu_layer.repack_weights()
    gpu_layer.repack_weights()
    if dp4a:
        gpu_layer.set_gpu_packed_inference(True)

    x_cpu = nsos.Tensor.random([batch, in_features], nsos.Device.CPU)
    x_gpu = x_cpu.to(nsos.Device.GPU)

    cpu = time_call(
        f"bitlinear[{batch}x{in_features}->{out_features}] cpu_packed",
        lambda: cpu_layer.forward(x_cpu),
        repeat=repeat,
    )
    gpu_label = (
        f"bitlinear[{batch}x{in_features}->{out_features}] "
        + ("gpu_dp4a" if dp4a else "gpu_float")
    )
    gpu = time_call(
        gpu_label,
        lambda: gpu_layer.forward(x_gpu).cpu(),
        repeat=repeat,
    )
    return {
        "op": "bitlinear_dp4a" if dp4a else "bitlinear_float",
        "shape": {"batch": batch, "in_features": in_features,
                   "out_features": out_features},
        "cpu": cpu,
        "gpu": gpu,
        "speedup_cpu_div_gpu": speedup(cpu["mean_s"], gpu["mean_s"]),
    }


def bench_mamba(nsos, d_model: int, d_state: int, n_heads: int, batch: int,
                seq: int, *, repeat: int) -> Dict[str, Any]:
    """Mamba2SSD full-sequence forward, exercises Phase 3 selective_scan."""
    cpu_layer = nsos.Mamba2SSD(d_model, d_state, n_heads)
    gpu_layer = nsos.Mamba2SSD(d_model, d_state, n_heads)

    cpu_params = cpu_layer.parameters()
    gpu_params = gpu_layer.parameters()
    for cp, gp in zip(cpu_params, gpu_params):
        gp.data.copy_from(cp.data)

    gpu_layer.to(nsos.Device.GPU)

    x_cpu = nsos.Tensor.random([batch, seq, d_model], nsos.Device.CPU)
    x_gpu = x_cpu.to(nsos.Device.GPU)

    cpu = time_call(
        f"mamba[{batch}x{seq}x{d_model}] cpu",
        lambda: cpu_layer.forward(x_cpu, None),
        repeat=repeat,
    )
    gpu = time_call(
        f"mamba[{batch}x{seq}x{d_model}] gpu_selective_scan",
        lambda: gpu_layer.forward(x_gpu, None).cpu(),
        repeat=repeat,
    )
    return {
        "op": "mamba_selective_scan",
        "shape": {"batch": batch, "seq": seq, "d_model": d_model,
                   "d_state": d_state, "n_heads": n_heads},
        "cpu": cpu,
        "gpu": gpu,
        "speedup_cpu_div_gpu": speedup(cpu["mean_s"], gpu["mean_s"]),
    }


def bench_moe_router(nsos, d_model: int, num_experts: int, top_k: int,
                     rows: int, *, repeat: int) -> Dict[str, Any]:
    """MoERouter forward, exercises Phase 4 GPU top-k mask + load accum."""
    cpu_router = nsos.MoERouter(d_model, num_experts, top_k)
    gpu_router = nsos.MoERouter(d_model, num_experts, top_k)

    cpu_params = cpu_router.parameters()
    gpu_params = gpu_router.parameters()
    for cp, gp in zip(cpu_params, gpu_params):
        gp.data.copy_from(cp.data)

    gpu_router.to(nsos.Device.GPU)

    x_cpu = nsos.Tensor.random([rows, d_model], nsos.Device.CPU)
    x_gpu = x_cpu.to(nsos.Device.GPU)

    cpu = time_call(
        f"moe_router[{rows}x{d_model}->{num_experts}@{top_k}] cpu",
        lambda: cpu_router.forward(x_cpu),
        repeat=repeat,
    )
    # MoERouter.forward returns (logits, weights) — pull weights to CPU
    # to make the GPU path actually sync.
    def gpu_call():
        _, w = gpu_router.forward(x_gpu)
        return w.cpu()
    gpu = time_call(
        f"moe_router[{rows}x{d_model}->{num_experts}@{top_k}] gpu_fastpath",
        gpu_call,
        repeat=repeat,
    )
    return {
        "op": "moe_router",
        "shape": {"rows": rows, "d_model": d_model,
                   "num_experts": num_experts, "top_k": top_k},
        "cpu": cpu,
        "gpu": gpu,
        "speedup_cpu_div_gpu": speedup(cpu["mean_s"], gpu["mean_s"]),
    }


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path,
                        help="Directory containing nsos_ext.pyd (CUDA build)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Per-op timed iterations (after warmup)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write JSON report")
    args = parser.parse_args()

    nsos = _load_nsos(args.build_dir)

    # Quick CUDA sanity check; bail out cleanly if no device visible.
    cuda_present = False
    try:
        # Allocating a 1-element GPU tensor will throw if CUDA missing.
        nsos.Tensor.zeros([1], nsos.Device.GPU)
        cuda_present = True
    except Exception as ex:
        print(f"[bench] CUDA not available: {ex}", file=sys.stderr)
        sys.exit(2)
    print(f"[bench] CUDA available: {cuda_present}")

    results: List[Dict[str, Any]] = []
    print("\n=== matmul ===")
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256),
                   (512, 512, 512)]:
        r = bench_matmul(nsos, *shape, repeat=args.repeat)
        results.append(r)
        print(f"  {shape}: cpu={r['cpu']['mean_s']*1e3:.2f}ms "
              f"gpu={r['gpu']['mean_s']*1e3:.2f}ms "
              f"speedup={r['speedup_cpu_div_gpu']:.2f}x")

    print("\n=== bitlinear ===")
    for cfg in [(64, 64, 4), (128, 128, 8), (256, 256, 16), (512, 512, 32)]:
        r = bench_bitlinear(nsos, *cfg, repeat=args.repeat, dp4a=False)
        results.append(r)
        print(f"  {cfg} float: cpu={r['cpu']['mean_s']*1e3:.2f}ms "
              f"gpu={r['gpu']['mean_s']*1e3:.2f}ms "
              f"speedup={r['speedup_cpu_div_gpu']:.2f}x")
        r = bench_bitlinear(nsos, *cfg, repeat=args.repeat, dp4a=True)
        results.append(r)
        print(f"  {cfg} dp4a:  cpu={r['cpu']['mean_s']*1e3:.2f}ms "
              f"gpu={r['gpu']['mean_s']*1e3:.2f}ms "
              f"speedup={r['speedup_cpu_div_gpu']:.2f}x")

    print("\n=== mamba ===")
    for cfg in [(64, 16, 4, 1, 32), (128, 16, 4, 1, 64), (256, 16, 4, 1, 128)]:
        r = bench_mamba(nsos, *cfg, repeat=args.repeat)
        results.append(r)
        print(f"  d_model={cfg[0]} seq={cfg[4]}: cpu={r['cpu']['mean_s']*1e3:.2f}ms "
              f"gpu={r['gpu']['mean_s']*1e3:.2f}ms "
              f"speedup={r['speedup_cpu_div_gpu']:.2f}x")

    print("\n=== moe_router ===")
    for cfg in [(64, 8, 2, 32), (128, 8, 2, 64), (256, 16, 2, 128)]:
        r = bench_moe_router(nsos, *cfg, repeat=args.repeat)
        results.append(r)
        print(f"  d_model={cfg[0]} experts={cfg[1]} rows={cfg[3]}: "
              f"cpu={r['cpu']['mean_s']*1e3:.2f}ms "
              f"gpu={r['gpu']['mean_s']*1e3:.2f}ms "
              f"speedup={r['speedup_cpu_div_gpu']:.2f}x")

    summary = {
        "device": "GTX 1050 Ti (sm_61, GTX 1050 Ti reference)",
        "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[bench] wrote report → {args.report}")


if __name__ == "__main__":
    main()
