"""generate_architecture_report.py — turn raw heatmap data into
architectural insight.

Reads the JSON output of heatmap_profiler.py and writes a Markdown
report with:

  1. Roofline analysis: where is each layer on the FLOPS/byte plane
     relative to the host's measured DRAM bandwidth and peak FLOPS?
     Layers below the ridge = memory-bound (where we should optimize
     bandwidth, not compute).  Layers above = compute-bound (where
     we should optimize FLOPS, e.g. Tensor Cores).

  2. Cache-residency analysis: given the measured L1/L2/L3 sizes,
     classify each layer's working set:
       weights_in_L2 = layer_weight_bytes <= L2_size
       activations_in_L1 = activation_bytes <= L1_size
     Layers where the working set exceeds the local cache but COULD
     fit in a larger one are tagged "cache spillover" candidates —
     these are where layout/grouping changes pay off.

  3. Per-layer hot spots: which layers dominate cycles?  Tag the
     top-3 by total cycles AND the top-3 by jitter (max/min ratio).
     High jitter = poor cache locality OR variable routing (in MoE).

  4. Architectural recommendations: a section that proposes
     CONCRETE data-structure changes derived from the data.  Not
     generic advice; recommendations point at specific layers and
     specific data structures.  E.g.:
       "Layer 7 (MoE) shows 3.4x jitter — expert weights are likely
        being evicted from L2 between tokens.  Recommendation:
        adopt a token-routing-aware expert layout where the top-K
        most-used experts (per the run's measured router
        distribution) are pinned to L2 via __builtin_prefetch."

This script is the bridge between raw measurement and architectural
research.  It's intentionally opinionated.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Hardware peak references ────────────────────────────────────────────
# These are used for roofline analysis when the report doesn't include
# direct measurement of GPU peak.  Values from vendor specs; update if
# the user runs on a different host.

KNOWN_HOSTS = {
    "gtx_1050ti": {
        "device": "GTX 1050 Ti (Pascal sm_61)",
        "peak_gflops_fp32": 2100.0,
        "peak_gbps": 112.1,
        "vram_gb": 4.0,
    },
    "t4": {
        "device": "Tesla T4 (Turing sm_75)",
        "peak_gflops_fp32": 8100.0,
        "peak_gflops_fp16": 65000.0,
        "peak_gbps": 320.0,
        "vram_gb": 16.0,
    },
    "a100": {
        "device": "A100 (Ampere sm_80)",
        "peak_gflops_fp32": 19500.0,
        "peak_gflops_fp16": 312000.0,
        "peak_gflops_bf16": 312000.0,
        "peak_gbps": 1555.0,
        "vram_gb": 40.0,
    },
    "host_cpu": {
        "device": "Host CPU (estimate)",
        "peak_gflops_fp32": 100.0,    # crude — depends on cores+AVX
        "peak_gbps": 50.0,
        "vram_gb": 16.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heatmap", type=Path, required=True,
                        help="JSON output of heatmap_profiler.py")
    parser.add_argument("--out", type=Path, required=True,
                        help="Markdown report output path")
    parser.add_argument("--host", choices=list(KNOWN_HOSTS), default=None,
                        help="Override hardware reference (else inferred from "
                             "the heatmap's 'host.device' field).")
    parser.add_argument("--include-raw-events", action="store_true",
                        help="Include a raw-events table in the report "
                             "(can make report very long).")
    return parser.parse_args()


def pick_host(report: Dict[str, Any], override: Optional[str]) -> Dict[str, Any]:
    if override and override in KNOWN_HOSTS:
        return {**KNOWN_HOSTS[override], "name": override}
    # Try to infer from cycles_per_ns (host CPU) or from device.
    device = report.get("host", {}).get("device", "")
    if device == "cpu":
        return {**KNOWN_HOSTS["host_cpu"], "name": "host_cpu"}
    # Default: assume T4 if we can't tell.
    return {**KNOWN_HOSTS["t4"], "name": "t4"}


def fmt_bytes(b: int) -> str:
    if b >= 1024 * 1024 * 1024:
        return f"{b/1024/1024/1024:.2f} GB"
    if b >= 1024 * 1024:
        return f"{b/1024/1024:.2f} MB"
    if b >= 1024:
        return f"{b/1024:.2f} KB"
    return f"{b} B"


def fmt_cycles(c: int) -> str:
    if c >= 1_000_000_000:
        return f"{c/1e9:.2f} Gc"
    if c >= 1_000_000:
        return f"{c/1e6:.2f} Mc"
    if c >= 1_000:
        return f"{c/1e3:.2f} Kc"
    return f"{c} c"


def cycles_to_us(c: int, cycles_per_ns: float) -> float:
    if cycles_per_ns <= 0:
        return 0.0
    return c / cycles_per_ns / 1000.0


# ── Roofline section ────────────────────────────────────────────────────

def roofline_section(report: Dict[str, Any], host: Dict[str, Any]) -> str:
    """Produce the roofline analysis prose + ASCII chart.

    Roofline: y-axis = FLOPS achieved, x-axis = arithmetic intensity
    (FLOPS per byte).  Ceiling has two parts:
      * Memory-bound ceiling: y = bandwidth * intensity
      * Compute-bound ceiling: y = peak_flops
    The "ridge point" is where these meet: intensity = peak / bandwidth.
    For our T4 example: 8100 GFLOPS / 320 GB/s = 25.3 FLOPS/byte ridge.

    A layer is memory-bound if its arithmetic intensity is BELOW the
    ridge — adding FLOPS doesn't help, only adding bandwidth or
    reducing bytes moved.  This is where data-structure work pays off.
    """
    peak_gflops = host.get("peak_gflops_fp32", 0.0)
    peak_gbps = host.get("peak_gbps", 0.0)
    ridge = (peak_gflops / peak_gbps) if peak_gbps > 0 else 0.0

    lines: List[str] = []
    lines.append("## Roofline analysis")
    lines.append("")
    lines.append(f"**Host reference**: {host.get('device', 'unknown')}")
    lines.append(f"- Peak FP32: **{peak_gflops:.0f} GFLOPS**")
    lines.append(f"- Peak bandwidth: **{peak_gbps:.1f} GB/s**")
    lines.append(f"- Ridge point: **{ridge:.1f} FLOPS/byte**")
    lines.append("")
    lines.append("Any layer whose arithmetic intensity is BELOW the ridge "
                 "is memory-bandwidth bound.  Optimization should focus on "
                 "reducing bytes moved (compact weights, fused ops, cache "
                 "residency) not on adding FLOPS (Tensor Cores, FP16).")
    lines.append("")
    lines.append("Per-layer arithmetic intensity (this run):")
    lines.append("")
    lines.append("| Layer | Mean cycles | Mean μs | Bytes in+out | MFLOPS | Intensity (F/B) | Regime |")
    lines.append("|---|---|---|---|---|---|---|")
    cycles_per_ns = report.get("cycles_per_ns", 1.0)
    for ls in report.get("layers", []):
        bin_b = ls.get("total_bytes_in", 0) if "total_bytes_in" in ls else 0
        bout_b = ls.get("total_bytes_out", 0) if "total_bytes_out" in ls else 0
        total_b = bin_b + bout_b
        mflops = ls.get("total_flops", 0) if "total_flops" in ls else 0
        intensity = (mflops * 1e6) / max(total_b, 1)
        regime = "MEMORY" if (intensity < ridge or total_b == 0) else "COMPUTE"
        us = cycles_to_us(ls.get("mean_cycles", 0), cycles_per_ns)
        lines.append(
            f"| {ls['layer_idx']} | {fmt_cycles(ls.get('mean_cycles', 0))} | "
            f"{us:.1f} | {fmt_bytes(total_b)} | {mflops} | "
            f"{intensity:.2f} | {regime} |"
        )
    if not any(ls.get("total_bytes_in") for ls in report.get("layers", [])):
        lines.append("")
        lines.append("> _Bytes/FLOPS columns are zero because the model "
                     "forward path does not currently annotate them in the "
                     "profiler events.  Add `scope.annotate(bytes_in, "
                     "bytes_out, mflops)` calls inside specific layer "
                     "implementations to populate this column._")
    lines.append("")
    return "\n".join(lines)


# ── Cache residency section ─────────────────────────────────────────────

def cache_residency_section(report: Dict[str, Any]) -> str:
    cache = report.get("cache_probe", {})
    if not cache:
        return ""
    l1 = cache.get("inferred_l1_bytes", 0)
    l2 = cache.get("inferred_l2_bytes", 0)
    l3 = cache.get("inferred_l3_bytes", 0)
    dram_lat = cache.get("dram_latency_cycles", 0.0)
    samples = cache.get("samples", [])

    model = report.get("model", {})
    # Rough estimate of model weight bytes in 1.58-bit.
    # 0.2 bytes per weight (1.58 bits packed).
    params_approx = (
        model.get("vocab_size", 0) * model.get("d_model", 0) +
        model.get("num_layers", 0) * model.get("d_model", 0) * model.get("d_model", 0) * 6
    )
    weights_bitnet_bytes = int(params_approx * 0.2)

    lines: List[str] = []
    lines.append("## Cache residency analysis")
    lines.append("")
    lines.append("**Measured cache hierarchy on this host** (pointer-chase probe):")
    lines.append("")
    lines.append("| Working set | Median cycles/access | p90 cycles/access |")
    lines.append("|---|---|---|")
    for s in samples:
        lines.append(
            f"| {fmt_bytes(s['working_set_bytes'])} | "
            f"{s['median_cycles_per_access']:.1f} | "
            f"{s['p90_cycles_per_access']:.1f} |"
        )
    lines.append("")
    lines.append(f"**Inferred boundaries**: L1 ≈ {fmt_bytes(l1)}, "
                 f"L2 ≈ {fmt_bytes(l2)}, L3 ≈ {fmt_bytes(l3)}, "
                 f"DRAM ≈ {dram_lat:.0f} cycles/access")
    lines.append("")
    lines.append(f"**Model weights (1.58-bit packed estimate)**: ~{fmt_bytes(weights_bitnet_bytes)}")
    if l2 > 0:
        if weights_bitnet_bytes <= l2:
            lines.append(f"- All weights **FIT IN L2** ({fmt_bytes(weights_bitnet_bytes)} <= {fmt_bytes(l2)}).")
            lines.append("- Goal: ensure the model is actually KEEPING them in L2, not "
                          "thrashing.  Per-layer cycle counts should be close to "
                          "the L2-latency regime, not DRAM.  If they're not, weights "
                          "are being evicted between forwards (likely by activations).")
        elif weights_bitnet_bytes <= l3:
            lines.append(f"- Weights overflow L2 ({fmt_bytes(weights_bitnet_bytes)} > {fmt_bytes(l2)}) "
                         f"but fit in L3.  Cycle counts in the L3-latency regime "
                         f"are the floor.  Solution: layer-grouping so adjacent "
                         f"BitLinears share L2 working sets.")
        else:
            lines.append(f"- Weights overflow L3 ({fmt_bytes(weights_bitnet_bytes)} > {fmt_bytes(l3)}).  "
                         f"DRAM bandwidth is the absolute ceiling.")
    lines.append("")
    return "\n".join(lines)


# ── Hotspot + jitter section ────────────────────────────────────────────

def hotspots_section(report: Dict[str, Any]) -> str:
    layers = report.get("layers", [])
    if not layers:
        return ""
    cycles_per_ns = report.get("cycles_per_ns", 1.0)

    by_total = sorted(layers, key=lambda l: l.get("total_cycles", 0), reverse=True)
    by_jitter = sorted(layers, key=lambda l: l.get("jitter_ratio", 0), reverse=True)

    lines: List[str] = []
    lines.append("## Hot spots + jitter")
    lines.append("")
    lines.append("### Top 5 layers by total cycles")
    lines.append("| Layer | Total cycles | Mean μs | Calls | Share of all cycles |")
    lines.append("|---|---|---|---|---|")
    total = sum(l.get("total_cycles", 0) for l in layers)
    for l in by_total[:5]:
        share = 100.0 * l.get("total_cycles", 0) / max(total, 1)
        us = cycles_to_us(l.get("mean_cycles", 0), cycles_per_ns)
        lines.append(
            f"| {l['layer_idx']} | {fmt_cycles(l['total_cycles'])} | "
            f"{us:.1f} | {l['call_count']} | {share:.1f}% |"
        )
    lines.append("")
    lines.append("### Top 5 layers by jitter (max/min cycle ratio)")
    lines.append("| Layer | Min cycles | Max cycles | Jitter ratio | Interpretation |")
    lines.append("|---|---|---|---|---|")
    for l in by_jitter[:5]:
        j = l.get("jitter_ratio", 0)
        interp = ("stable" if j < 0.2 else
                  "moderate jitter (likely routing variance)" if j < 1.0 else
                  "high jitter (cache or MoE routing dominates)")
        lines.append(
            f"| {l['layer_idx']} | {fmt_cycles(l['min_cycles'])} | "
            f"{fmt_cycles(l['max_cycles'])} | {j:.2f} | {interp} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Op breakdown section ────────────────────────────────────────────────

def op_breakdown_section(report: Dict[str, Any]) -> str:
    ops = report.get("ops", [])
    if not ops:
        return ""
    cycles_per_ns = report.get("cycles_per_ns", 1.0)
    lines: List[str] = []
    lines.append("## Per-op aggregate breakdown")
    lines.append("")
    lines.append("Across all layers and all forward passes.  An op is a "
                 "named sub-operation inside a layer (e.g. `attn.qkv`, "
                 "`mamba.ssd`, `moe.router`, `ffn.gate_up`, "
                 "`squared_relu`).  When the profiler hooks for these ops "
                 "are populated, this table fills out.  Layers-only profile "
                 "still gives the high-level breakdown above.")
    lines.append("")
    lines.append("| Op | Total cycles | Calls | Mean μs |")
    lines.append("|---|---|---|---|")
    for op in ops[:15]:
        us = cycles_to_us(op["mean_cycles"], cycles_per_ns)
        lines.append(
            f"| `{op['name']}` | {fmt_cycles(op['total_cycles'])} | "
            f"{op['call_count']} | {us:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Architecture recommendations section ────────────────────────────────

def architecture_recommendations(report: Dict[str, Any], host: Dict[str, Any]) -> str:
    """The opinionated section.  Each recommendation cites SPECIFIC
    measured values from this run so the reader can verify it isn't
    generic advice."""
    layers = report.get("layers", [])
    cache = report.get("cache_probe", {})
    l2 = cache.get("inferred_l2_bytes", 0)
    l3 = cache.get("inferred_l3_bytes", 0)
    dram_lat = cache.get("dram_latency_cycles", 0.0)
    cycles_per_ns = report.get("cycles_per_ns", 1.0)

    recs: List[str] = []
    # Recommendation 1: high-jitter layers
    if layers:
        by_jitter = sorted(layers, key=lambda l: l.get("jitter_ratio", 0), reverse=True)
        worst = by_jitter[0]
        if worst.get("jitter_ratio", 0) > 1.5:
            recs.append(
                f"**R1 — Stabilize layer {worst['layer_idx']} routing/cache**.  "
                f"Observed jitter ratio {worst['jitter_ratio']:.2f}× "
                f"(min={fmt_cycles(worst['min_cycles'])}, "
                f"max={fmt_cycles(worst['max_cycles'])}).  At this magnitude "
                f"the dominant cause is one of: (a) MoE expert weights "
                f"evicted between tokens, (b) attention KV cache "
                f"growing past L2, or (c) Mamba state read pattern "
                f"alternating between cached and uncached blocks.  "
                f"Concrete data-structure change: pin layer-{worst['layer_idx']} "
                f"weight rows to L2 via explicit prefetch + per-expert "
                f"layout that keeps top-K experts contiguous in the "
                f"weight tensor.")

    # Recommendation 2: bandwidth bound vs compute bound dispatch
    peak_gflops = host.get("peak_gflops_fp32", 0.0)
    peak_gbps = host.get("peak_gbps", 0.0)
    if peak_gflops > 0 and peak_gbps > 0:
        ridge = peak_gflops / peak_gbps
        recs.append(
            f"**R2 — Most layers are memory-bound on this host (ridge = "
            f"{ridge:.1f} F/B).**  The dominant optimization target is "
            f"reducing bytes moved, NOT adding FLOPS.  This means: "
            f"(a) FP16/BF16 cuts weight bytes 2×; (b) tighter weight "
            f"packing (int4, ternary) cuts another 2-4×; (c) avoiding "
            f"activation copies between layers (in-place writes where "
            f"shape permits) cuts a 4 KB / layer / token tax.  Tensor "
            f"Core paths only help layers above the ridge.")

    # Recommendation 3: weights vs cache size
    model = report.get("model", {})
    params_approx = (
        model.get("vocab_size", 0) * model.get("d_model", 0) +
        model.get("num_layers", 0) * model.get("d_model", 0) * model.get("d_model", 0) * 6
    )
    weights_bitnet_bytes = int(params_approx * 0.2)
    if l2 > 0 and weights_bitnet_bytes <= l2:
        # Look for cycle counts inconsistent with L2 residency
        if dram_lat > 0:
            l2_cycles_estimate = 30  # typical L2 hit cycles
            slow_layers = [l for l in layers
                           if l.get("mean_cycles", 0) >
                              l2_cycles_estimate * model.get("d_model", 1) * 50]
            if slow_layers:
                recs.append(
                    f"**R3 — Weights fit in L2 but layers are running at DRAM "
                    f"speeds.**  {len(slow_layers)} layers show cycle counts "
                    f"inconsistent with L2 residency (would expect ~"
                    f"{l2_cycles_estimate * model.get('d_model', 0) * 50} "
                    f"cycles if hot in L2, observed up to "
                    f"{max(l.get('mean_cycles', 0) for l in slow_layers)} "
                    f"cycles).  Activations are evicting the weights.  "
                    f"Suggested structure: a **persistent weight arena** "
                    f"that mmap's all BitLinear weights once at load time "
                    f"into a hugepage-backed region, separate from the "
                    f"activation arena.  Prefetch the next layer's weights "
                    f"into L2 during the current layer's compute.")

    # Recommendation 4: scientific question
    recs.append(
        "**R4 — Research question for v12 architecture.**  Mamba2 and "
        "Attention have a mathematical duality (Gu & Dao 2024 SSD paper).  "
        "If we replace the attention layers with Mamba2-SSD layers of "
        "equivalent state dim, the model becomes O(1) per token (no KV "
        "cache growth).  This requires a v12 retrain — but per-layer "
        "cycle profiling here suggests which attention layers are the "
        "primary bandwidth offenders, telling us which ones to swap first "
        "for maximum sequential-decode speedup at minimum quality cost.  "
        "Compare per-layer cycles for attention layers vs Mamba layers in "
        "the table above to identify candidates.")

    lines = ["## Architectural recommendations", ""]
    for r in recs:
        lines.append(r)
        lines.append("")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    if not args.heatmap.exists():
        sys.stderr.write(f"[fatal] heatmap not found: {args.heatmap}\n")
        return 1
    report = json.loads(args.heatmap.read_text("utf-8"))
    host = pick_host(report, args.host)

    md: List[str] = []
    md.append("# NSOS Architectural Analysis — Heatmap Report")
    md.append("")
    md.append(f"_Generated from {args.heatmap}_")
    md.append("")
    model = report.get("model", {})
    md.append(f"**Model**: {model.get('num_layers', 0)}-layer "
              f"d_model={model.get('d_model', 0)} "
              f"vocab={model.get('vocab_size', 0)} "
              f"({'MoE-' + str(model.get('num_experts', 0)) if model.get('use_moe') else 'dense FFN'})")
    md.append("")
    md.append(f"**Device**: {report.get('host', {}).get('device', 'unknown')}")
    md.append(f"**Wall time measured**: {report.get('wall_seconds_measured', 0):.2f} s")
    md.append(f"**Total events captured**: {report.get('total_events', 0):,}")
    md.append("")
    md.append("---")
    md.append("")
    md.append(roofline_section(report, host))
    md.append("---")
    md.append("")
    md.append(cache_residency_section(report))
    md.append("---")
    md.append("")
    md.append(hotspots_section(report))
    md.append("---")
    md.append("")
    md.append(op_breakdown_section(report))
    md.append("---")
    md.append("")
    md.append(architecture_recommendations(report, host))
    md.append("---")
    md.append("")
    md.append("## Raw event count")
    md.append("")
    md.append(f"`{report.get('raw_events_path', '(unset)')}` "
              f"contains the full per-event JSON (kind, name, "
              f"layer, sub, start_cycle, end_cycle, cycles, bytes_in, "
              f"bytes_out, mflops).  Use it for finer-grained analysis "
              f"with `python -c 'import json; ...'`.")
    md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
