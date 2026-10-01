"""analyze_chrass_ab.py — agregar reports do probe_chrass_ab.py.

Lê todos os probe_chrass_d{N}_s{seed}.json no diretório, computa:
  - Loss curves overlaid (PNG)
  - Tabela: final loss, loss reduction, ms/step, NaN count
  - Effect size (Cohen's d) por density vs baseline (0%)
  - Recomendação: melhor density por effect size + estabilidade

Uso:
    python analyze_chrass_ab.py --reports-dir probe_reports/
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def load_reports(d: Path) -> Dict[int, Dict]:
    """Return {density: report}, taking only the first seed if multiple."""
    out: Dict[int, Dict] = {}
    for f in sorted(d.glob("probe_chrass_d*_s*.json")):
        try:
            r = json.loads(f.read_text())
            density = int(r["density"])
            if density not in out:
                out[density] = r
        except Exception as exc:
            print(f"  skipping {f.name}: {exc}")
    return out


def cohens_d_window(a: List[float], b: List[float]) -> float:
    """Standardized mean diff between two loss windows.  Negative = a is lower."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = statistics.pstdev(a)
    sb = statistics.pstdev(b)
    pooled = math.sqrt((sa * sa + sb * sb) / 2.0)
    if pooled < 1e-9:
        return 0.0
    return (ma - mb) / pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=Path("probe_reports"))
    ap.add_argument("--window", type=int, default=200,
                    help="Window of final steps to compare per density")
    ap.add_argument("--plot", action="store_true",
                    help="Generate loss-curve PNG (requires matplotlib)")
    args = ap.parse_args()

    reports = load_reports(args.reports_dir)
    if not reports:
        print(f"No reports found in {args.reports_dir}")
        return 1

    densities = sorted(reports.keys())
    print(f"Loaded {len(reports)} reports: densities = {densities}")

    if 0 not in reports:
        print("WARNING: no baseline (density=0) report found; can't compute effect size")
        baseline = None
    else:
        baseline = reports[0]
        baseline_window = baseline["loss"]["all"][-args.window:]
        baseline_final = statistics.fmean(baseline_window) if baseline_window else float("nan")
        print(f"  baseline (d=0) final {args.window}-step mean loss: {baseline_final:.4f}")

    # ── Per-density table
    print("\n┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ density  │ init_50  │ final-100│ min_loss │ ms/step  │ Cohen d  │")
    print("│   (%)    │          │          │          │          │ vs base  │")
    print("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    rows = []
    for d in densities:
        r = reports[d]
        all_losses = r["loss"]["all"]
        init = statistics.fmean(all_losses[:50]) if len(all_losses) >= 50 else float("nan")
        final = statistics.fmean(all_losses[-100:]) if len(all_losses) >= 100 else float("nan")
        mn = min(all_losses) if all_losses else float("nan")
        msps = r["timing"]["ms_per_step"]
        if baseline is not None:
            d_eff = cohens_d_window(
                all_losses[-args.window:],
                baseline["loss"]["all"][-args.window:],
            )
        else:
            d_eff = float("nan")
        rows.append({
            "density": d, "init": init, "final": final, "min": mn,
            "msps": msps, "d_eff": d_eff,
        })
        print(f"│   {d:>4d}   │  {init:6.3f}  │  {final:6.3f}  │  {mn:6.3f}  │  {msps:6.1f}  │  {d_eff:+6.3f}  │")
    print("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # ── Recommendation
    if baseline is not None:
        # Best density = lowest final loss (or most negative Cohen's d vs baseline)
        non_baseline_rows = [r for r in rows if r["density"] != 0]
        if non_baseline_rows:
            best_loss = min(non_baseline_rows, key=lambda r: r["final"])
            best_d = min(non_baseline_rows, key=lambda r: r["d_eff"])
            print()
            print(f"  RECOMMENDATION based on this A/B probe:")
            print(f"    lowest final loss: density={best_loss['density']}% (loss={best_loss['final']:.4f})")
            print(f"    largest improvement vs baseline: density={best_d['density']}% (d={best_d['d_eff']:+.3f})")
            print()
            # Interpretation
            if best_d['d_eff'] < -0.5:
                print(f"    -> CHRASS at density={best_d['density']}% shows MEANINGFUL IMPROVEMENT (Cohen's d < -0.5)")
            elif best_d['d_eff'] < -0.2:
                print(f"    -> CHRASS at density={best_d['density']}% shows SMALL improvement")
            elif best_d['d_eff'] > 0.2:
                print(f"    -> CHRASS HURTS at all tested densities; recommend OFF")
            else:
                print(f"    -> CHRASS effect is NEGLIGIBLE; not worth the ~4% overhead")

    # ── Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            for d in densities:
                losses = reports[d]["loss"]["all"]
                # smoothed
                k = 20
                smooth = [statistics.fmean(losses[max(0, i - k):i + 1])
                          for i in range(len(losses))]
                lab = f"d={d}%" if d > 0 else "baseline"
                ax.plot(range(len(smooth)), smooth, label=lab, alpha=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss (smoothed)")
            ax.set_title("CHRASS A/B Probe — Loss curves by density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            png_path = args.reports_dir / "chrass_ab_curves.png"
            plt.savefig(png_path, dpi=120, bbox_inches="tight")
            print(f"\n  plot saved -> {png_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
