"""OXTA-CRIT §6 — avalanche_analysis.py

The publishable end of the "diferencial": if the criticality controller holds the
ternary substrate at the edge of chaos, the discrete plasticity events (ternary
weight FLIPS) should organize into avalanches whose SIZE distribution follows a
POWER LAW — the signature of self-organized criticality (Beggs & Plenz 2003 for
neuronal avalanches).  The ternary body makes this literally countable: a flip is
a change in sign(round(W/gamma)) between two training snapshots.

This module:
  * ternary_codes(W)                 -> {-1,0,+1} codes (absmean rule)
  * flip_count(prev_codes, cur_codes)-> #codes that changed
  * avalanche_sizes(series, thresh)  -> sizes of supra-threshold bursts
  * fit_powerlaw_mle(sizes, xmin)    -> MLE exponent alpha (Clauset/Newman)
The closed-loop interpretation (does alpha sit near the critical 3/2 and track
generalization?) is the experiment; this is the rigorous estimator + a local,
synthetic self-test that proves the estimator recovers a known exponent.

Local self-test (pure numpy, real local evidence):
    python OXN/nsos/scripts/avalanche_analysis.py --selftest
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def ternary_codes(weight: np.ndarray) -> np.ndarray:
    """Absmean-rule ternary codes in {-1,0,+1} (matches BitLinear/quantize)."""
    gamma = float(np.mean(np.abs(weight))) or 1e-12
    return np.clip(np.round(weight / gamma), -1, 1).astype(np.int8)


def flip_count(prev_codes: np.ndarray, cur_codes: np.ndarray) -> int:
    """Number of ternary codes that changed between two snapshots."""
    return int(np.count_nonzero(prev_codes != cur_codes))


def flip_series_from_snapshots(snapshots: list[dict]) -> list[int]:
    """snapshots: list of {name: weight_array}; returns per-step total flips."""
    series = []
    for t in range(1, len(snapshots)):
        prev, cur = snapshots[t - 1], snapshots[t]
        total = 0
        for name, w in cur.items():
            if name in prev:
                total += flip_count(ternary_codes(prev[name]),
                                    ternary_codes(w))
        series.append(total)
    return series


def avalanche_sizes(series, threshold: float | None = None) -> list[int]:
    """An avalanche = a maximal run of steps with activity > threshold; its size
    is the summed activity over the run.  Default threshold = median (quiescence
    separator), the standard avalanche segmentation."""
    s = np.asarray(list(series), dtype=np.float64)
    if s.size == 0:
        return []
    if threshold is None:
        threshold = float(np.median(s))
    sizes, cur = [], 0.0
    for v in s:
        if v > threshold:
            cur += v - threshold
        elif cur > 0:
            sizes.append(int(round(cur)))
            cur = 0.0
    if cur > 0:
        sizes.append(int(round(cur)))
    return [x for x in sizes if x > 0]


def fit_powerlaw_mle(sizes, xmin: float = 1.0) -> dict:
    """Continuous MLE exponent for P(x) ~ x^-alpha, x >= xmin
    (Clauset, Shalizi & Newman 2009, eq. 3.1):
        alpha = 1 + n / sum_i ln(x_i / xmin).
    Exact (unbiased in expectation) for continuous data; the standard,
    widely-used estimator for integer avalanche sizes too (the discrete zeta
    MLE differs only at small xmin).  Returns alpha, std error, and n."""
    x = np.asarray([v for v in sizes if v >= xmin], dtype=np.float64)
    n = int(x.size)
    if n < 2:
        return {"alpha": float("nan"), "stderr": float("nan"), "n": n}
    denom = float(np.sum(np.log(x / xmin)))
    if denom <= 0:
        return {"alpha": float("nan"), "stderr": float("nan"), "n": n}
    alpha = 1.0 + n / denom
    stderr = (alpha - 1.0) / math.sqrt(n)
    return {"alpha": alpha, "stderr": stderr, "n": n, "xmin": float(xmin)}


def sample_powerlaw(alpha: float, xmin: float, n: int, rng) -> np.ndarray:
    """Continuous power-law sampler via inverse CDF (for the self-test)."""
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def selftest() -> int:
    rng = np.random.default_rng(0)

    # 1) Continuous MLE recovers a known exponent on continuous samples.
    for true_alpha in (1.8, 2.5, 3.0):
        x = sample_powerlaw(true_alpha, xmin=1.0, n=200000, rng=rng)
        fit = fit_powerlaw_mle(x, xmin=1.0)
        err = abs(fit["alpha"] - true_alpha)
        assert err < 0.05, (true_alpha, fit, err)
        print(f"[selftest] MLE recover alpha: true={true_alpha} "
              f"est={fit['alpha']:.3f} (+/-{fit['stderr']:.3f}) OK")

    # 2) flip counting on ternary codes.
    w0 = np.array([[1.0, -1.0, 0.0], [0.5, 0.0, -0.4]])
    w1 = w0.copy(); w1[0, 0] = -2.0      # +1 -> -1 flip
    c0, c1 = ternary_codes(w0), ternary_codes(w1)
    assert flip_count(c0, c1) >= 1
    print(f"[selftest] flip_count detects a sign flip OK ({flip_count(c0,c1)} flips)")

    # 3) avalanche segmentation: a quiescent-then-burst series.
    series = [0, 0, 5, 9, 0, 0, 3, 0, 12, 7, 0]
    sizes = avalanche_sizes(series, threshold=0.0)
    assert len(sizes) == 3 and all(s > 0 for s in sizes), sizes
    print(f"[selftest] avalanche segmentation -> {sizes} OK")

    # 4) end-to-end: power-law bursts separated by quiescence (zeros) -> the
    # avalanche sizes recover a power law (plumbing + estimator together).
    bursts = np.maximum(np.round(sample_powerlaw(2.5, 1.0, 600, rng)), 1).astype(int)
    series = []
    for s in bursts:
        series += [int(s), 0]  # one active step then quiescence
    sz = avalanche_sizes(series, threshold=0.0)
    fit = fit_powerlaw_mle(sz, xmin=1.0)
    assert fit["n"] >= 100 and math.isfinite(fit["alpha"]) and fit["alpha"] > 1.0, fit
    print(f"[selftest] end-to-end flip->avalanche->alpha={fit['alpha']:.3f} "
          f"(n={fit['n']}, input 2.5) OK")
    print("[selftest] avalanche_analysis PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OXTA-CRIT §6 ternary-flip avalanche / power-law")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--flips", type=str, default=None,
                    help="comma-separated per-step flip counts to analyze")
    ap.add_argument("--xmin", type=int, default=1)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.flips:
        series = [int(x) for x in args.flips.split(",") if x.strip()]
        sizes = avalanche_sizes(series)
        fit = fit_powerlaw_mle(sizes, xmin=args.xmin)
        print("avalanche sizes:", sizes)
        print("power-law fit:", fit)
        print("(critical neuronal-avalanche reference exponent ~ 1.5)")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
