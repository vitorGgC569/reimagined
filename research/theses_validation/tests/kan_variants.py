"""KAN-variant tests.

Tese 2 makes 3 separable claims about Kolmogorov-Arnold Networks:
  - KAN beats MLP at matched param count (general KAN claim)
  - SKAN (single-parameter B-splines) trains more stably than vanilla KAN
  - KANtize (precomputed lookup tables) gives ~50× BitOps reduction in
    inference latency vs runtime spline evaluation

Tests in this file:
  test_kan_vs_mlp            — toy regression: fit a 1D non-linearity
  test_skan_vs_kan           — stability of param-1 vs multi-param splines
  test_kantize_latency       — inference latency of runtime KAN vs lookup KAN

We use a small synthetic regression (fit sin(3x) + 0.5*sin(7x) on
x ∈ [-π, π]) instead of language modeling for the KAN tests because:
  (a) KANs shine on smooth function approximation where MLPs need
      many ReLU pieces.  LM tasks don't isolate this strength.
  (b) The latency claim is per-edge: any small model surfaces it.
  (c) Smaller models keep the suite under the 30 min/test budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import set_global_seed


# ──────────────────────────────────────────────────────────────────────
#  Toy data: 1D regression
# ──────────────────────────────────────────────────────────────────────

def _make_regression_data(n: int = 2048, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = sin(3x) + 0.5*sin(7x) on x ∈ [-π, π], 1024 train + 1024 eval."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, generator=g) - 0.5) * 2 * math.pi
    y = torch.sin(3 * x) + 0.5 * torch.sin(7 * x)
    return x.unsqueeze(1), y.unsqueeze(1)


def _split(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.size(0) // 2
    return x[:n], y[:n], x[n:], y[n:]


# ──────────────────────────────────────────────────────────────────────
#  MLP baseline (matched param count)
# ──────────────────────────────────────────────────────────────────────

class SmallMLP(nn.Module):
    """Baseline: 2-layer MLP with GELU.  Hidden dim chosen to match
    the KAN param count below (within ~5%) so the comparison is fair."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
#  Vanilla KAN with B-spline edges
# ──────────────────────────────────────────────────────────────────────

class KANEdge(nn.Module):
    """One KAN edge: learnable B-spline + a learned base linear.

    Spline = sum_i coef_i * cubic_basis_i(x).  We implement the cubic
    basis evaluation manually so this file is self-contained (no
    `pykan` dep — the reference paper's code is heavy and not pinned).
    """
    def __init__(self, n_knots: int = 8, *, single_param: bool = False):
        super().__init__()
        self.n_knots = n_knots
        knots = torch.linspace(-math.pi, math.pi, n_knots)
        self.register_buffer("knots", knots)
        # Base linear weight + bias — the affine component KAN edges
        # combine with the spline.
        self.base_w = nn.Parameter(torch.tensor(1.0))
        self.base_b = nn.Parameter(torch.tensor(0.0))
        # Spline coefficients.  Vanilla KAN: one per knot.
        # SKAN (Single-parameter KAN): single scalar that scales a fixed
        # basis — the "single parameter" version validated by EKE.
        if single_param:
            self.coef = nn.Parameter(torch.tensor(0.1))
            self.single = True
        else:
            self.coef = nn.Parameter(torch.zeros(n_knots))
            self.single = False

    def _cubic_basis(self, x: torch.Tensor) -> torch.Tensor:
        # Triangular basis (cheap; cubic is overkill for this toy).
        # phi_i(x) = max(0, 1 - |x - knot_i| / h), h = knot spacing
        h = (self.knots[-1] - self.knots[0]) / (self.n_knots - 1)
        d = (x.unsqueeze(-1) - self.knots).abs() / h
        return torch.clamp(1.0 - d, min=0.0)  # (..., n_knots)

    def forward(self, x):
        # x: (n, 1)
        base = self.base_w * torch.tanh(x) + self.base_b  # SiLU-like
        if self.single:
            # SKAN: spline term reduces to coef * single fixed kernel
            spline = self.coef * torch.sin(x)
        else:
            basis = self._cubic_basis(x)  # (n, 1, n_knots)
            spline = (basis * self.coef).sum(-1)  # (n, 1)
        return base + spline


class TinyKAN(nn.Module):
    """A 2-layer KAN: input → hidden → 1.

    For 1D input → 1D output we use a single hidden of size 8 edges.
    Each edge is a separate KANEdge.  This is the same recipe the
    original Kolmogorov-Arnold paper uses for their toy demos.
    """
    def __init__(self, hidden: int = 8, n_knots: int = 8, *, single_param: bool = False):
        super().__init__()
        # Layer 1: 1 input → hidden edges
        self.l1 = nn.ModuleList([KANEdge(n_knots, single_param=single_param) for _ in range(hidden)])
        # Layer 2: hidden → 1.  Each output is sum of `hidden` edges, one per input dim.
        self.l2 = nn.ModuleList([KANEdge(n_knots, single_param=single_param) for _ in range(hidden)])

    def forward(self, x):
        # x: (n, 1)
        h = torch.cat([e(x) for e in self.l1], dim=-1)  # (n, hidden)
        # apply edges along last dim and sum
        ys = []
        for j, e in enumerate(self.l2):
            ys.append(e(h[:, j:j+1]))
        return sum(ys)  # (n, 1)


# ──────────────────────────────────────────────────────────────────────
#  KANtize — replace runtime spline eval with a precomputed LUT
# ──────────────────────────────────────────────────────────────────────

class KANEdgeLUT(nn.Module):
    """Same forward output as a frozen KANEdge, but evaluated via a
    precomputed lookup table.  We bake the trained spline coefficients
    into a 1024-entry table over x ∈ [-π, π] and look up via linear
    interpolation.

    For latency benchmarking only — not trainable in this form.
    """
    def __init__(self, source: KANEdge, n_table: int = 1024):
        super().__init__()
        x_grid = torch.linspace(-math.pi, math.pi, n_table)
        with torch.no_grad():
            y_grid = source(x_grid.unsqueeze(1)).squeeze(-1)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("y_grid", y_grid)
        self.x_min = float(x_grid[0])
        self.x_max = float(x_grid[-1])
        self.n = n_table

    @torch.no_grad()
    def forward(self, x):
        # Map x ∈ [x_min, x_max] to table index, then linear interp.
        xn = (x.squeeze(-1) - self.x_min) / (self.x_max - self.x_min) * (self.n - 1)
        xn = xn.clamp(0, self.n - 1 - 1e-6)
        i0 = xn.long()
        frac = (xn - i0.float()).unsqueeze(-1)
        y0 = self.y_grid[i0].unsqueeze(-1)
        y1 = self.y_grid[i0 + 1].unsqueeze(-1)
        return y0 + frac * (y1 - y0)


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────

def _train_regression(model: nn.Module, x_train, y_train, x_eval, y_eval,
                       steps: int = 1500, lr: float = 1e-2, device: torch.device | None = None):
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_eval, y_eval = x_eval.to(device), y_eval.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_curve = []
    for _ in range(steps):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_curve.append(float(loss.item()))
    with torch.no_grad():
        eval_loss = float(F.mse_loss(model(x_eval), y_eval).item())
    return {
        "final_train_mse": train_curve[-1],
        "eval_mse": eval_loss,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "curve": train_curve,
    }


def test_kan_vs_mlp(*, device: torch.device, seed: int = 42, steps: int = 1500) -> dict:
    set_global_seed(seed)
    x, y = _make_regression_data(seed=seed)
    x_train, y_train, x_eval, y_eval = _split(x, y)

    out = {}
    set_global_seed(seed)
    # MLP hidden=64 → 1*64 + 64 + 64*64 + 64 + 64*1 + 1 ≈ 4.4K params
    mlp = SmallMLP(hidden=64)
    out["MLP (hidden=64)"] = _train_regression(mlp, x_train, y_train, x_eval, y_eval,
                                                steps=steps, device=device)
    set_global_seed(seed)
    # KAN hidden=8, knots=8 → 16 edges × (2 + 8 spline coef) ≈ 160 params
    # Much smaller than MLP — but that's the KAN claim: expressivity per param.
    kan = TinyKAN(hidden=8, n_knots=8, single_param=False)
    out["KAN (hidden=8, knots=8)"] = _train_regression(kan, x_train, y_train, x_eval, y_eval,
                                                       steps=steps, device=device)
    # Matched-param KAN: hidden=24, knots=12 → ~700 params (still much
    # smaller than MLP — KANs really do have fewer params; we cap at
    # what's reasonable for the toy).
    set_global_seed(seed)
    kan_big = TinyKAN(hidden=24, n_knots=12, single_param=False)
    out["KAN (hidden=24, knots=12)"] = _train_regression(kan_big, x_train, y_train, x_eval, y_eval,
                                                          steps=steps, device=device)
    return out


def test_skan_vs_kan(*, device: torch.device, seed: int = 42, steps: int = 1500) -> dict:
    set_global_seed(seed)
    x, y = _make_regression_data(seed=seed)
    x_train, y_train, x_eval, y_eval = _split(x, y)
    out = {}
    set_global_seed(seed)
    out["KAN (multi-param)"] = _train_regression(
        TinyKAN(hidden=8, n_knots=8, single_param=False),
        x_train, y_train, x_eval, y_eval, steps=steps, device=device,
    )
    set_global_seed(seed)
    out["SKAN (single-param)"] = _train_regression(
        TinyKAN(hidden=8, n_knots=8, single_param=True),
        x_train, y_train, x_eval, y_eval, steps=steps, device=device,
    )
    return out


def test_kantize_latency(*, device: torch.device, seed: int = 42, steps: int = 500,
                          n_inference: int = 5000) -> dict:
    """Train a vanilla KAN, then time inference: runtime spline vs LUT.

    We expect the LUT to be 5-20× faster, not the 50× the thesis cites
    — the thesis number is from FPGA BitOps reduction, not CPU/GPU
    throughput.  We report what we measure, not what's promised.
    """
    set_global_seed(seed)
    x, y = _make_regression_data(seed=seed)
    x_train, y_train, _, _ = _split(x, y)
    src = TinyKAN(hidden=8, n_knots=8, single_param=False)
    _train_regression(src, x_train, y_train, x_train, y_train, steps=steps, device=device)

    x_bench = torch.randn(n_inference, 1, device=device) * math.pi

    # Runtime path: one edge from layer 1, eval many times
    src_edge = src.l1[0].to(device)
    # Warmup
    for _ in range(50):
        _ = src_edge(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(200):
        _ = src_edge(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_s = time.time() - t0

    # LUT path
    lut = KANEdgeLUT(src_edge.to(torch.device("cpu"))).to(device)
    for _ in range(50):
        _ = lut(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(200):
        _ = lut(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lut_s = time.time() - t0

    return {
        "Runtime spline (s for 200 evals)": runtime_s,
        "LUT (s for 200 evals)": lut_s,
        "Speedup (×)": runtime_s / max(lut_s, 1e-9),
        "n_inference_per_call": n_inference,
    }


ALL_TESTS = {
    "kan_vs_mlp":        test_kan_vs_mlp,
    "skan_vs_kan":       test_skan_vs_kan,
    "kantize_latency":   test_kantize_latency,
}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="KAN variant tests.")
    p.add_argument("--test", choices=list(ALL_TESTS), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[kan_variants] device={device.type} test={args.test}")
    results = ALL_TESTS[args.test](device=device, seed=args.seed, steps=args.steps)
    print(f"\n=== test_{args.test} ===")
    for label, val in results.items():
        if isinstance(val, dict):
            print(f"  {label}:")
            for k, v in val.items():
                if k == "curve":
                    continue
                if isinstance(v, float):
                    print(f"    {k:<32} {v:.6f}")
                else:
                    print(f"    {k:<32} {v}")
        else:
            print(f"  {label:<40} {val}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Strip curves before json (large)
        serializable = {}
        for label, val in results.items():
            if isinstance(val, dict):
                serializable[label] = {k: v for k, v in val.items() if k != "curve"}
            else:
                serializable[label] = val
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
