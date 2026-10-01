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
Honest: warmup and measured samples are reported separately; if GPU is slower
than CPU for a small shape, the report says so.

Run:
  $env:PYTHONIOENCODING = "utf-8"
  python bench_gpu_vs_cpu.py --build-dir <path-to-configured-gpu-build>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from oxta_contabil.benchmark_policy import configured_test_inventory

import numpy as np


_dll_handles: list[Any] = []


def _load_nsos(build_dir: Path):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        # Keep handles alive for the lifetime of the extension.
        candidates = [
            build_dir,
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"),
        ]
        for key in ("NSOS_HIP_ROOT", "ROCM_PATH", "HIP_PATH"):
            if os.environ.get(key):
                root = Path(os.environ[key])
                candidates.extend(
                    (root / "bin", root / "lib" / "llvm" / "bin")
                )
        for path in candidates:
            try:
                if path.exists():
                    _dll_handles.append(
                        os.add_dll_directory(str(path))
                    )
            except AttributeError as exc:
                raise RuntimeError(
                    "Python on Windows does not expose os.add_dll_directory"
                ) from exc
            except OSError as exc:
                print(
                    f"[runtime] DLL directory rejected: {path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
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
        raise RuntimeError("measured GPU latency must be positive")
    return cpu_mean / gpu_mean


def parity_metrics(
    cpu_tensor,
    gpu_tensor,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    cpu = np.asarray(cpu_tensor.numpy(), dtype=np.float32)
    gpu = np.asarray(gpu_tensor.numpy(), dtype=np.float32)
    if cpu.shape != gpu.shape:
        raise RuntimeError(
            f"CPU/GPU parity shape mismatch: {cpu.shape} != {gpu.shape}"
        )
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise RuntimeError("CPU/GPU parity produced a non-finite value")
    absolute = np.abs(cpu - gpu)
    maximum = float(absolute.max(initial=0.0))
    mean = float(absolute.mean()) if absolute.size else 0.0
    if not np.allclose(cpu, gpu, atol=atol, rtol=rtol):
        flat_index = int(np.argmax(absolute))
        index = np.unravel_index(flat_index, absolute.shape)
        raise RuntimeError(
            "CPU/GPU parity failed at "
            f"{index}: cpu={float(cpu[index])} gpu={float(gpu[index])} "
            f"abs={float(absolute[index])} atol={atol} rtol={rtol}"
        )
    return {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "elements": int(absolute.size),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────
# Individual benches
# ────────────────────────────────────────────────────────────────────

def bench_matmul(nsos, M: int, K: int, N: int, *, repeat: int) -> Dict[str, Any]:
    a_cpu = nsos.Tensor.random([M, K], nsos.Device.CPU)
    b_cpu = nsos.Tensor.random([K, N], nsos.Device.CPU)
    a_gpu = a_cpu.to(nsos.Device.GPU)
    b_gpu = b_cpu.to(nsos.Device.GPU)
    parity = parity_metrics(
        a_cpu.matmul(b_cpu),
        a_gpu.matmul(b_gpu).cpu(),
        atol=1e-3,
        rtol=2e-4,
    )

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
        "parity": parity,
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
        gp.copy_data_from(cp.data)

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
    parity = parity_metrics(
        cpu_layer.forward(x_cpu),
        gpu_layer.forward(x_gpu).cpu(),
        atol=3e-3,
        rtol=5e-4,
    )

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
        "parity": parity,
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
        gp.copy_data_from(cp.data)

    gpu_layer.to(nsos.Device.GPU)

    x_cpu = nsos.Tensor.random([batch, seq, d_model], nsos.Device.CPU)
    x_gpu = x_cpu.to(nsos.Device.GPU)
    parity = parity_metrics(
        cpu_layer.forward(x_cpu, None),
        gpu_layer.forward(x_gpu, None).cpu(),
        atol=3e-3,
        rtol=5e-4,
    )

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
        "parity": parity,
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
        gp.copy_data_from(cp.data)

    gpu_router.to(nsos.Device.GPU)

    x_cpu = nsos.Tensor.random([rows, d_model], nsos.Device.CPU)
    x_gpu = x_cpu.to(nsos.Device.GPU)
    cpu_logits, cpu_weights = cpu_router.forward(x_cpu)
    gpu_logits, gpu_weights = gpu_router.forward(x_gpu)
    parity = {
        "logits": parity_metrics(
            cpu_logits,
            gpu_logits.cpu(),
            atol=2e-3,
            rtol=5e-4,
        ),
        "weights": parity_metrics(
            cpu_weights,
            gpu_weights.cpu(),
            atol=2e-3,
            rtol=5e-4,
        ),
    }

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
        "parity": parity,
        "speedup_cpu_div_gpu": speedup(cpu["mean_s"], gpu["mean_s"]),
    }


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path,
                        help="Configured directory containing nsos_ext")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Per-op timed iterations (after warmup)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write JSON report")
    args = parser.parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    nsos = _load_nsos(args.build_dir)
    test_inventory = configured_test_inventory(args.build_dir)

    backend = nsos.gpu_backend_name()
    if backend not in {"cuda", "hip"}:
        raise RuntimeError(
            f"benchmark requires a CUDA or HIP build, got {backend!r}"
        )
    devices = list(nsos.gpu_devices())
    selected_device = int(nsos.selected_gpu_device())
    selected = next(
        (
            device
            for device in devices
            if int(device["index"]) == selected_device
        ),
        None,
    )
    if selected is None or not bool(selected.get("compiled", False)):
        raise RuntimeError(
            "selected GPU is absent or not compiled into this binary"
        )
    if test_inventory["gpu_backend"].lower() != backend:
        raise RuntimeError(
            "configured CTest inventory backend does not match the loaded "
            "NSOS extension"
        )
    nsos.set_strict_gpu_execution(True)
    try:
        nsos.Tensor.zeros([1], nsos.Device.GPU)
    except Exception as ex:
        print(f"[bench] GPU not available: {ex}", file=sys.stderr)
        sys.exit(2)
    print(
        f"[bench] backend={backend} selected_device={selected}",
        flush=True,
    )

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
        "schema_version": 2,
        "backend": backend,
        "selected_device": selected,
        "visible_devices": devices,
        "configured_test_inventory": test_inventory,
        "strict_gpu_execution": bool(nsos.strict_gpu_execution()),
        "repeat": args.repeat,
        "warmup": 5,
        "timing_scope": "operation plus synchronized output transfer to CPU",
        "results": results,
    }
    if args.report:
        atomic_write_json(args.report, summary)
        print(f"\n[bench] wrote report → {args.report}")


if __name__ == "__main__":
    main()
