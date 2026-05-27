"""analyze_cascade.py — agregar reports do probe_cascade.py.

Espera ler probe_*.json no diretório.  Computa:
  - Tabela: label, params, ms/step, loss init, loss final, % melhoria vs baseline
  - Cohen's d de cada tech vs baseline
  - Veredito por tech: HELP / NEUTRO / HURT
  - Plot loss curves overlaid (PNG)
  - Summary JSON
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from typing import Dict, List, Optional


def load_reports(d: Path) -> Dict[str, Dict]:
    out = {}
    for f in sorted(d.glob("probe_*.json")):
        try:
            r = json.loads(f.read_text())
            out[r["label"]] = r
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return out


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa, sb = statistics.pstdev(a), statistics.pstdev(b)
    pooled = math.sqrt((sa*sa + sb*sb) / 2.0)
    return 0.0 if pooled < 1e-9 else (ma - mb) / pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=Path("cascade_reports"))
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    reports = load_reports(args.reports_dir)
    if not reports:
        print(f"No reports in {args.reports_dir}")
        return 1
    print(f"Loaded {len(reports)} reports: {list(reports.keys())}")

    baseline = reports.get(args.baseline_label)
    if baseline:
        base_w = baseline["loss"]["all"][-args.window:]
        base_final = statistics.fmean(base_w) if base_w else float("nan")
        base_msps = baseline["timing"]["ms_per_step"]
        print(f"\nBaseline: final_loss={base_final:.4f}, ms/step={base_msps:.1f}")

    print("\n┌─────────────────┬────────┬────────┬────────┬─────────┬──────────┬─────────┐")
    print("│ label           │  init  │ final  │  min   │ ms/step │ Δ vs base│ Cohen d │")
    print("├─────────────────┼────────┼────────┼────────┼─────────┼──────────┼─────────┤")

    rows = []
    for lbl, r in reports.items():
        all_l = r["loss"]["all"]
        init = statistics.fmean(all_l[:50]) if len(all_l) >= 50 else float("nan")
        final = statistics.fmean(all_l[-100:]) if len(all_l) >= 100 else float("nan")
        mn = min(all_l) if all_l else float("nan")
        msps = r["timing"]["ms_per_step"]
        if baseline and lbl != args.baseline_label:
            w = all_l[-args.window:]
            d = cohens_d(w, base_w)
            delta = (final - base_final) / max(abs(base_final), 1e-6) * 100
        else:
            d = 0.0; delta = 0.0
        rows.append({"label": lbl, "init": init, "final": final,
                     "min": mn, "msps": msps, "delta_pct": delta, "d": d})
        delta_str = f"{delta:+5.1f}%" if lbl != args.baseline_label else "  ref  "
        d_str = f"{d:+6.3f}" if lbl != args.baseline_label else "  ref  "
        print(f"│ {lbl:<15} │ {init:6.3f} │ {final:6.3f} │ {mn:6.3f} │ {msps:7.1f} │ {delta_str} │ {d_str} │")

    print("└─────────────────┴────────┴────────┴────────┴─────────┴──────────┴─────────┘")

    if baseline:
        print("\nPer-tech veredict (negative d/delta = improvement vs baseline):")
        for r in rows:
            if r["label"] == args.baseline_label: continue
            if r["d"] < -0.5: v = "✓ MEANINGFUL IMPROVEMENT"
            elif r["d"] < -0.2: v = "✓ small improvement"
            elif r["d"] > 0.5: v = "✗ MEANINGFUL DEGRADATION"
            elif r["d"] > 0.2: v = "⚠ small degradation"
            else: v = "= neutral (within noise)"
            print(f"  {r['label']:<15}: {v}")

    summary = {
        "baseline": args.baseline_label,
        "window": args.window,
        "rows": rows,
    }
    out_summary = args.reports_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out_summary}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            for lbl, r in reports.items():
                losses = r["loss"]["all"]
                k = 20
                smooth = [statistics.fmean(losses[max(0,i-k):i+1]) for i in range(len(losses))]
                lw = 2.5 if lbl == args.baseline_label else 1.5
                style = "-" if lbl == args.baseline_label else "--"
                ax.plot(smooth, label=lbl, linewidth=lw, linestyle=style, alpha=0.85)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss (smoothed)")
            ax.set_title("Oxta cascade — loss curves by tech config")
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            p = args.reports_dir / "cascade_curves.png"
            plt.savefig(p, dpi=120, bbox_inches="tight")
            print(f"Plot -> {p}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
