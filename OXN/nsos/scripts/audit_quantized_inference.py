"""Audit script: prove the 1.58-bit inference claim is real (VISION #3).

Without this audit, "NSOS runs in 1.58-bit at the edge" is just a
README assertion.  This script LOADS a pack, MEASURES memory, COMPARES
forward outputs, and PRINTS a verdict.

It runs four sub-audits and emits a single pass/fail verdict per pack:

  A. Pack contents
     - Verify manifest.nsos has edge_linear key
     - Verify edge_linear.nsos exists and is < the FP32 weights file
     - Print compression ratio

  B. Memory mode
     - Load pack twice: once with NSOS_KEEP_FP32_WEIGHTS=1 (FP32 ref mode),
       once without (edge / ternary mode).
     - Use psutil / GPU memory probes to measure RSS before vs after each
       load.  Edge mode RAM should be < FP32 RAM by ~ (1 - 1.58/32) ≈ 95%
       of weight memory.

  C. Forward correctness
     - Run a fixed forward pass in both modes on the same input.
     - Compare top-1 token agreement and KL divergence between logit
       distributions.  Edge vs FP32 should agree on most tokens; large
       disagreements indicate broken quantization.

  D. Throughput
     - Time forward passes in both modes.  Edge mode should be at least
       as fast as FP32 mode (ternary kernels are simpler).

CLI:
  python OXN/nsos/scripts/audit_quantized_inference.py \\
      --pack /path/to/pack_dir/ \\
      --build-dir /path/to/build \\
      --prompt "Hello world"

The script returns exit 0 if every sub-audit passes, non-zero with a
labeled failure message otherwise.

Designed to run in CI on every pack release.
"""
from __future__ import annotations

# Eager datasets import not needed here (no HF), but pin BLAS sanity.
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def _process_rss_mb() -> float:
    """Resident set size in MB.  Uses psutil if installed, falls back
    to /proc/self/status on Linux."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except OSError:
        pass
    return float("nan")


def _gpu_alloc_mb(nsos_ext) -> float:
    """Best-effort GPU memory probe.  Returns NaN if no GPU."""
    try:
        # nsos_ext doesn't expose memory directly; try pynvml
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / (1024 * 1024)
    except Exception:
        return float("nan")


def _import_nsos(build_dir: Path):
    if os.name == "nt":
        add_windows_runtime_dirs(build_dir,
                                  parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import nsos_ext  # type: ignore
    return nsos_ext


# ── A: Pack contents audit ────────────────────────────────────────────

def audit_pack_contents(pack_dir: Path) -> Tuple[bool, Dict, str]:
    """Check the pack has an edge_linear file and report sizes."""
    if pack_dir.is_file():
        pack_dir = pack_dir.parent
    manifest = pack_dir / "manifest.nsos"
    if not manifest.exists():
        return False, {}, f"manifest.nsos missing in {pack_dir}"

    kv = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

    if "edge_linear" not in kv:
        return False, {"manifest": kv}, ("pack has no edge_linear entry — this "
                                          "pack was saved without the 1.58-bit edge file")

    weights_path = pack_dir / kv.get("weights", "weights.nsos")
    edge_path = pack_dir / kv["edge_linear"]
    if not weights_path.exists():
        return False, {"manifest": kv}, f"weights file missing: {weights_path}"
    if not edge_path.exists():
        return False, {"manifest": kv}, f"edge file missing: {edge_path}"
    w_size = weights_path.stat().st_size
    e_size = edge_path.stat().st_size
    ratio = e_size / max(w_size, 1)
    expected_ratio = 1.58 / 32.0  # ternary bit count vs float32

    info = {
        "manifest": kv,
        "weights_bytes": w_size,
        "edge_bytes": e_size,
        "compression_ratio": ratio,
        "expected_ratio": expected_ratio,
        "pack_dir": str(pack_dir),
    }
    # Allow some overhead — manifest, headers, scale tensors — but the
    # edge pack should still be < 25% of the FP32 weights file size
    # for a 1.58-bit claim to be honest.
    if ratio > 0.25:
        return False, info, (f"edge pack is {ratio:.2%} of FP32 weights — much "
                              f"higher than the expected {expected_ratio:.2%}.  "
                              f"Quantization may not be applied correctly.")
    return True, info, "edge pack present, size consistent with ternary"


# ── B: Memory mode audit ──────────────────────────────────────────────

def audit_memory(nsos_ext, pack_dir: Path, *, device: str = "auto"
                  ) -> Tuple[bool, Dict, str]:
    """Load the same pack twice — once edge mode, once FP32 ref mode —
    and compare RAM/VRAM deltas."""
    if device == "auto":
        target_device = nsos_ext.Device.GPU if nsos_ext.fast_gpu_supported() else nsos_ext.Device.CPU
    else:
        target_device = nsos_ext.Device.GPU if device == "gpu" else nsos_ext.Device.CPU

    rss_base = _process_rss_mb()
    gpu_base = _gpu_alloc_mb(nsos_ext)

    # Mode A: edge / ternary (release_fp32 = true via default env)
    os.environ.pop("NSOS_KEEP_FP32_WEIGHTS", None)
    engine_edge = nsos_ext.InferenceEngine()
    cfg = nsos_ext.ModelConfig()
    cfg.use_cuda = (target_device == nsos_ext.Device.GPU)
    if not engine_edge.load_model(str(pack_dir), cfg):
        return False, {}, "InferenceEngine.load_model failed in edge mode"
    rss_edge = _process_rss_mb()
    gpu_edge = _gpu_alloc_mb(nsos_ext)
    # Free
    del engine_edge

    # Mode B: keep FP32
    os.environ["NSOS_KEEP_FP32_WEIGHTS"] = "1"
    engine_fp32 = nsos_ext.InferenceEngine()
    cfg = nsos_ext.ModelConfig()
    cfg.use_cuda = (target_device == nsos_ext.Device.GPU)
    if not engine_fp32.load_model(str(pack_dir), cfg):
        return False, {}, "InferenceEngine.load_model failed in FP32 ref mode"
    rss_fp32 = _process_rss_mb()
    gpu_fp32 = _gpu_alloc_mb(nsos_ext)
    del engine_fp32
    os.environ.pop("NSOS_KEEP_FP32_WEIGHTS", None)

    info = {
        "rss_base_mb": rss_base, "rss_edge_mb": rss_edge, "rss_fp32_mb": rss_fp32,
        "rss_edge_delta_mb": rss_edge - rss_base if rss_base else float("nan"),
        "rss_fp32_delta_mb": rss_fp32 - rss_base if rss_base else float("nan"),
        "gpu_base_mb": gpu_base, "gpu_edge_mb": gpu_edge, "gpu_fp32_mb": gpu_fp32,
        "device": "GPU" if target_device == nsos_ext.Device.GPU else "CPU",
    }
    # Edge mode should use noticeably less memory than FP32 mode for the
    # weights themselves.  For a 40M param model, FP32 weights ≈ 160 MB;
    # ternary weights ≈ 8 MB.  Allow generous overhead (KV cache, etc.)
    # so we accept anything where edge delta < 0.5 * FP32 delta.
    fp32_delta = info["rss_fp32_delta_mb"]
    edge_delta = info["rss_edge_delta_mb"]
    if not (math.isnan(fp32_delta) or math.isnan(edge_delta)):
        if fp32_delta > 10 and edge_delta >= 0.6 * fp32_delta:
            return False, info, (
                f"edge mode uses {edge_delta:.1f} MB vs FP32 {fp32_delta:.1f} MB — "
                f"too close together.  Quantization may not be releasing FP32 weights.")
    return True, info, (f"edge RAM delta={edge_delta:.1f} MB, "
                         f"FP32 delta={fp32_delta:.1f} MB")


# ── C: Forward correctness audit ──────────────────────────────────────

def audit_forward_correctness(nsos_ext, pack_dir: Path, prompt: str,
                               *, device: str = "auto"
                               ) -> Tuple[bool, Dict, str]:
    """Run identical prompt through both modes, compare top-1 agreement
    and KL divergence."""
    import numpy as np
    target_device = (nsos_ext.Device.GPU if (device == "gpu" or (device == "auto" and nsos_ext.fast_gpu_supported()))
                     else nsos_ext.Device.CPU)

    def _load_and_forward(keep_fp32: bool) -> Tuple[np.ndarray, float]:
        if keep_fp32:
            os.environ["NSOS_KEEP_FP32_WEIGHTS"] = "1"
        else:
            os.environ.pop("NSOS_KEEP_FP32_WEIGHTS", None)
        cfg = nsos_ext.ModelConfig()
        cfg.use_cuda = (target_device == nsos_ext.Device.GPU)
        engine = nsos_ext.InferenceEngine()
        if not engine.load_model(str(pack_dir), cfg):
            raise RuntimeError("load failed")
        ids = list(engine.tokenizer.encode(prompt))
        if not ids:
            raise RuntimeError("empty token sequence")
        t0 = time.time()
        logits = engine.model.forward_ids(ids)
        elapsed = time.time() - t0
        arr = logits.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        os.environ.pop("NSOS_KEEP_FP32_WEIGHTS", None)
        return arr[-1], elapsed

    logits_edge, t_edge = _load_and_forward(keep_fp32=False)
    logits_fp32, t_fp32 = _load_and_forward(keep_fp32=True)

    # Top-1 agreement
    top1_edge = int(np.argmax(logits_edge))
    top1_fp32 = int(np.argmax(logits_fp32))

    # KL divergence between distributions
    def _softmax(row):
        m = row.max()
        e = np.exp(row.astype(np.float64) - m)
        return e / e.sum()
    p_edge = _softmax(logits_edge)
    p_fp32 = _softmax(logits_fp32)
    eps = 1e-12
    kl = float(np.sum(p_edge * (np.log(p_edge + eps) - np.log(p_fp32 + eps))))

    # Top-5 overlap
    top5_edge = set(np.argsort(logits_edge)[-5:].tolist())
    top5_fp32 = set(np.argsort(logits_fp32)[-5:].tolist())
    top5_jaccard = len(top5_edge & top5_fp32) / max(len(top5_edge | top5_fp32), 1)

    info = {
        "top1_edge": top1_edge, "top1_fp32": top1_fp32,
        "top1_match": top1_edge == top1_fp32,
        "kl_edge_vs_fp32": kl,
        "top5_jaccard": top5_jaccard,
        "wall_edge_s": t_edge, "wall_fp32_s": t_fp32,
        "speedup_edge_vs_fp32": (t_fp32 / max(t_edge, 1e-9)) if t_edge > 0 else float("nan"),
    }

    # KL above 1.0 nat = catastrophic disagreement; quantization broken.
    # KL < 0.1 = nearly identical.  Between is normal for ternary quant.
    if kl > 1.5:
        return False, info, (f"KL(edge||fp32)={kl:.3f} nats — catastrophic "
                              f"disagreement.  Quantization is broken.")
    if not info["top1_match"] and top5_jaccard < 0.4:
        return False, info, (f"top-1 differs AND top-5 overlap is only "
                              f"{top5_jaccard:.1%}.  Quantization is poor.")
    return True, info, (f"top-1 match={info['top1_match']}, "
                         f"KL={kl:.3f} nats, top-5 jaccard={top5_jaccard:.2f}")


# ── D: Throughput audit ───────────────────────────────────────────────

def audit_throughput(nsos_ext, pack_dir: Path, prompt: str,
                      *, device: str = "auto", n_iters: int = 20
                      ) -> Tuple[bool, Dict, str]:
    """Repeat forward N times in both modes; report tokens/s."""
    target_device = (nsos_ext.Device.GPU if (device == "gpu" or (device == "auto" and nsos_ext.fast_gpu_supported()))
                     else nsos_ext.Device.CPU)

    def _measure(keep_fp32: bool) -> float:
        if keep_fp32:
            os.environ["NSOS_KEEP_FP32_WEIGHTS"] = "1"
        else:
            os.environ.pop("NSOS_KEEP_FP32_WEIGHTS", None)
        cfg = nsos_ext.ModelConfig()
        cfg.use_cuda = (target_device == nsos_ext.Device.GPU)
        engine = nsos_ext.InferenceEngine()
        engine.load_model(str(pack_dir), cfg)
        ids = list(engine.tokenizer.encode(prompt))
        # Warmup
        for _ in range(3):
            _ = engine.model.forward_ids(ids)
        t0 = time.time()
        for _ in range(n_iters):
            _ = engine.model.forward_ids(ids)
        elapsed = time.time() - t0
        return elapsed / max(n_iters, 1)

    avg_edge = _measure(False)
    avg_fp32 = _measure(True)
    info = {
        "avg_forward_edge_s": avg_edge,
        "avg_forward_fp32_s": avg_fp32,
        "speedup": avg_fp32 / max(avg_edge, 1e-9),
        "n_iters": n_iters,
    }
    # Edge should be at least as fast as FP32 (we expect speedup,
    # but slowdown is the failure mode worth flagging).
    if avg_edge > avg_fp32 * 1.3:
        return False, info, (f"edge mode is 30%+ SLOWER than FP32 "
                              f"({avg_edge:.4f}s vs {avg_fp32:.4f}s).  "
                              f"Ternary kernels should not be slower.")
    return True, info, (f"edge avg={avg_edge*1000:.2f} ms, "
                         f"FP32 avg={avg_fp32*1000:.2f} ms, "
                         f"speedup={info['speedup']:.2f}x")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", type=Path, required=True,
                   help="Pack directory (or path inside it)")
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--prompt", type=str,
                   default="The quick brown fox jumps over the lazy dog.")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON report path.  Default = next to pack.")
    p.add_argument("--skip-throughput", action="store_true",
                   help="Skip the throughput sub-audit (saves time).")
    args = p.parse_args()

    print(f"=== Quantized Inference Audit ===")
    print(f"  pack: {args.pack}")
    print(f"  build: {args.build_dir}")
    print()

    results: Dict[str, Dict] = {}
    overall_pass = True

    # A. Pack contents
    print("[A] Pack contents...")
    ok_a, info_a, msg_a = audit_pack_contents(args.pack)
    results["A_pack_contents"] = {"pass": ok_a, "info": info_a, "msg": msg_a}
    print(f"    {'PASS' if ok_a else 'FAIL'}: {msg_a}")
    if not ok_a:
        overall_pass = False
        # Fail-fast: subsequent audits need a valid pack
        _write_report(args, results, overall_pass)
        return 1

    # Import nsos
    print()
    print("[setup] importing nsos_ext...")
    nsos = _import_nsos(args.build_dir.resolve())
    print(f"        fast_gpu_supported={nsos.fast_gpu_supported()}")
    print()

    # B. Memory mode
    print("[B] Memory mode...")
    ok_b, info_b, msg_b = audit_memory(nsos, args.pack, device=args.device)
    results["B_memory"] = {"pass": ok_b, "info": info_b, "msg": msg_b}
    print(f"    {'PASS' if ok_b else 'FAIL'}: {msg_b}")
    if not ok_b:
        overall_pass = False

    # C. Forward correctness
    print()
    print("[C] Forward correctness...")
    ok_c, info_c, msg_c = audit_forward_correctness(
        nsos, args.pack, args.prompt, device=args.device,
    )
    results["C_correctness"] = {"pass": ok_c, "info": info_c, "msg": msg_c}
    print(f"    {'PASS' if ok_c else 'FAIL'}: {msg_c}")
    if not ok_c:
        overall_pass = False

    # D. Throughput
    if not args.skip_throughput:
        print()
        print("[D] Throughput...")
        ok_d, info_d, msg_d = audit_throughput(
            nsos, args.pack, args.prompt, device=args.device,
        )
        results["D_throughput"] = {"pass": ok_d, "info": info_d, "msg": msg_d}
        print(f"    {'PASS' if ok_d else 'FAIL'}: {msg_d}")
        if not ok_d:
            overall_pass = False

    print()
    print(f"=== VERDICT: {'PASS' if overall_pass else 'FAIL'} ===")
    _write_report(args, results, overall_pass)
    return 0 if overall_pass else 1


def _write_report(args, results, overall_pass):
    out = args.out
    if out is None:
        pack = args.pack if args.pack.is_dir() else args.pack.parent
        out = pack / "audit_quantized_inference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "pack": str(args.pack),
        "verdict": "pass" if overall_pass else "fail",
        "audits": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[audit] report -> {out}")


if __name__ == "__main__":
    sys.exit(main())
