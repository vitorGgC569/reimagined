"""Quantization-estimator tests.

Tests in this file (each is an isolated A/B with one variable swapped):

  test_decoupled_ste_vs_ste       (Tese 1, section "Decoupled ST")
  test_hgf                        (Tese 1, "Hybrid Gated Flow")
  test_continual_qat              (Tese 1, "Pré-treinamento Ciente de Quantização Contínua")
  test_denoising_dequant          (Tese 2, "Denoising Dequantization Transform / Ridge")

Backbone: small Transformer + ternary linear projections inside
attention/FFN.  Same backbone, same seed, same batches — only the
gradient estimator for the ternary weights changes.

Why a Transformer and not RNN: the user's original POC used a hand-
rolled RNN-tanh which made the baseline pathological (untrained Linear
+ round(clamp) collapses to all-zero weights at init scale 0.02).  A
small Transformer with proper init and LayerNorm doesn't suffer this
init collapse, so STE vs alternatives is a fair comparison of
*estimators*, not "broken init vs working init".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make `common` importable when run as `python -m tests.quantization`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import load_wikitext_chars, set_global_seed
from common.training import train_and_eval, RunResult


# ──────────────────────────────────────────────────────────────────────
#  Gradient estimators (the variable under test)
# ──────────────────────────────────────────────────────────────────────

def _ternarize(weight: torch.Tensor) -> torch.Tensor:
    """Standard BitNet b1.58 quantization.

    Scale weights by mean absolute value, round to {-1, 0, +1}, scale back.
    This is the forward used by ALL the estimators below — what changes
    is only the gradient that flows BACKWARD through this op.
    """
    scale = weight.abs().mean().clamp_min(1e-8)
    return torch.round(torch.clamp(weight / scale, -1.0, 1.0)) * scale


class _STE(torch.autograd.Function):
    """Classical Straight-Through Estimator: forward quantizes, backward
    passes the gradient through as if quantization was the identity.

    grad_input = grad_output (identity)

    This is the baseline.  The pathology the theses describe is real
    but mild on properly-initialized weights: when weight is far from
    {-1, 0, +1} the gradient pushes it correctly; when weight is near
    a boundary (e.g. 0.001) the rounded forward decides one way but
    the gradient is computed as if the forward were continuous, which
    can cause oscillations.
    """
    @staticmethod
    def forward(ctx, weight):
        return _ternarize(weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _DecoupledSTE(torch.autograd.Function):
    """Decoupled STE (Tian et al. 2024 style): forward uses τ_f for
    discretization, backward uses τ_b for the gradient surrogate.

    The thesis describes the backward as proportional to the softmax
    Jacobian under τ_b.  For a 3-way ternary state this reduces to a
    smooth surrogate around the boundaries (1 - tanh²(w/τ_b)) — the
    tanh saturates fast enough that gradient magnitudes are bounded
    even when weights drift large, mitigating the oscillation that
    plain STE causes.
    """
    @staticmethod
    def forward(ctx, weight, tau_f, tau_b):
        ctx.save_for_backward(weight)
        ctx.tau_b = tau_b
        # Forward unchanged in scale; tau_f only affects which side of the
        # round() each weight lands on (sharper = more decisive).  We
        # implement that by scaling the input to round, then unscaling.
        scaled = weight / max(tau_f, 1e-6)
        return _ternarize(scaled) * max(tau_f, 1e-6)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        tau_b = max(ctx.tau_b, 1e-6)
        # Smooth surrogate: 1 - tanh²(w/τ_b) is the derivative of tanh(w/τ_b)
        # — a bounded function that's ~1 near zero (so we don't kill the
        # signal for small weights) and decays toward 0 for large |w|
        # (so we don't keep pushing weights that are already saturated).
        surrogate = (1.0 - torch.tanh(weight / tau_b).pow(2)) / tau_b
        return grad_output * surrogate, None, None


class _DenoisingDequant(torch.autograd.Function):
    """Denoising Dequantization Transform (Tese 2, section 3).

    The thesis says: model the quantization as additive noise η and
    write the backward via Ridge regression — gradient ≈ identity STE
    plus a Ridge penalty that pulls weights toward the nearest
    quantization grid point.

    Our implementation: backward = grad_output - λ * (weight - quantized_weight).
    The Ridge term shrinks weights toward their rounded value, which
    discourages drift across the {-1, 0, +1} boundaries that STE
    triggers.  λ defaults to 0.01 (small — should not dominate the
    gradient signal, just nudge).
    """
    @staticmethod
    def forward(ctx, weight, ridge_lambda):
        q = _ternarize(weight)
        ctx.save_for_backward(weight, q)
        ctx.ridge = ridge_lambda
        return q

    @staticmethod
    def backward(ctx, grad_output):
        (weight, q) = ctx.saved_tensors
        # The (weight - q) term is the dequantization noise; Ridge
        # subtracts a fraction of it from the upstream grad to push
        # weights toward their rounded targets.
        ridge_grad = ctx.ridge * (weight - q)
        return grad_output - ridge_grad, None


# ──────────────────────────────────────────────────────────────────────
#  Linear layers parametrized by estimator
# ──────────────────────────────────────────────────────────────────────

class TernaryLinear(nn.Module):
    """Linear layer where weight is ternarized via a chosen estimator.

    Bias stays FP32 always (only the weight matrix is quantized) —
    matches BitNet b1.58 spec.
    """
    def __init__(self, in_features: int, out_features: int, *,
                 estimator: str = "ste", tau_f: float = 1.0, tau_b: float = 0.5,
                 ridge_lambda: float = 0.01,
                 hgf: bool = False, hgf_init: float = 0.05,
                 bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.estimator = estimator
        self.tau_f = tau_f
        self.tau_b = tau_b
        self.ridge_lambda = ridge_lambda
        # Hybrid Gated Flow: parallel FP residual w * gate, gate per-output-channel
        self.hgf = hgf
        if hgf:
            self.hgf_gate = nn.Parameter(torch.full((out_features,), hgf_init))

    def _quantized_weight(self) -> torch.Tensor:
        if self.estimator == "ste":
            return _STE.apply(self.weight)
        elif self.estimator == "decoupled_ste":
            return _DecoupledSTE.apply(self.weight, self.tau_f, self.tau_b)
        elif self.estimator == "denoising_dequant":
            return _DenoisingDequant.apply(self.weight, self.ridge_lambda)
        elif self.estimator == "fp_baseline":
            # No quantization — for tests that compare quantized variants
            # against a no-quantization control.
            return self.weight
        else:
            raise ValueError(f"unknown estimator: {self.estimator}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._quantized_weight()
        out = F.linear(x, w, self.bias)
        if self.hgf:
            # FP residual gated per-output-channel.  Designed to be small
            # at init (0.05) so it doesn't dominate the ternary signal,
            # but can be learned upward where it stabilizes training.
            fp = F.linear(x, self.weight * self.hgf_gate.unsqueeze(1))
            out = out + fp
        return out


# ──────────────────────────────────────────────────────────────────────
#  Backbone: small Transformer where ALL Linear are TernaryLinear
# ──────────────────────────────────────────────────────────────────────

class TernaryTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, estimator: str,
                 tau_f: float, tau_b: float, ridge_lambda: float,
                 hgf: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        common_kw = dict(estimator=estimator, tau_f=tau_f, tau_b=tau_b,
                         ridge_lambda=ridge_lambda, hgf=hgf)
        self.q_proj = TernaryLinear(d_model, d_model, **common_kw)
        self.k_proj = TernaryLinear(d_model, d_model, **common_kw)
        self.v_proj = TernaryLinear(d_model, d_model, **common_kw)
        self.o_proj = TernaryLinear(d_model, d_model, **common_kw)
        self.ff1 = TernaryLinear(d_model, 4 * d_model, **common_kw)
        self.ff2 = TernaryLinear(4 * d_model, d_model, **common_kw)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # Causal scaled-dot-product (sdpa with is_causal=True is fine here)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.o_proj(attn)
        h = self.ln2(x)
        x = x + self.ff2(F.gelu(self.ff1(h)))
        return x


class TernaryLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_seq: int = 256, *,
                 estimator: str = "ste", tau_f: float = 1.0, tau_b: float = 0.5,
                 ridge_lambda: float = 0.01, hgf: bool = False):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.max_seq = max_seq
        self.blocks = nn.ModuleList([
            TernaryTransformerBlock(d_model, n_heads,
                                    estimator=estimator, tau_f=tau_f, tau_b=tau_b,
                                    ridge_lambda=ridge_lambda, hgf=hgf)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Head is FP linear — common practice in BitNet; ternary logits
        # would compress the softmax distribution too aggressively.
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        assert t <= self.max_seq
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
#  Test entry points
# ──────────────────────────────────────────────────────────────────────

def _build_model(vocab_size: int, **kw) -> TernaryLM:
    return TernaryLM(vocab_size, d_model=128, n_heads=4, n_layers=2, **kw)


def test_decoupled_ste_vs_ste(*, device: torch.device, dataset, steps: int = 1500,
                              seed: int = 42) -> dict:
    """STE vs Decoupled STE on identical TernaryLM backbone.

    Same seed → same init for weights AND same data order.  The ONLY
    difference is which autograd.Function provides the gradient through
    the round() in TernaryLinear.
    """
    out = {}
    for label, estimator in [("STE", "ste"), ("Decoupled-STE", "decoupled_ste")]:
        set_global_seed(seed)
        model = _build_model(dataset.vocab_size, estimator=estimator, hgf=False)
        res = train_and_eval(
            model, dataset, label=label, device=device,
            steps=steps, batch_size=32, seq_len=64, lr=3e-4,
            notes=f"estimator={estimator}",
        )
        out[label] = res
    return out


def test_hgf(*, device: torch.device, dataset, steps: int = 1500, seed: int = 42) -> dict:
    """Ternary-only vs Ternary + HGF (FP residual bypass).

    Both use STE so the only variable is the presence of the FP gate.
    This is the test the original POC failed: the user's POC had HGF
    only on one side, so it looked like Decoupled-STE was the winner
    when actually HGF (FP bypass) was carrying the network.  Here both
    use STE and we measure whether HGF actually stabilizes.
    """
    out = {}
    for label, hgf in [("Ternary-only", False), ("Ternary+HGF", True)]:
        set_global_seed(seed)
        model = _build_model(dataset.vocab_size, estimator="ste", hgf=hgf)
        res = train_and_eval(
            model, dataset, label=label, device=device,
            steps=steps, batch_size=32, seq_len=64, lr=3e-4,
            notes=f"hgf={hgf}",
        )
        out[label] = res
    return out


def test_continual_qat(*, device: torch.device, dataset, steps: int = 2000,
                       seed: int = 42, warmup_frac: float = 0.4) -> dict:
    """Cold-start ternary vs FP→ternary annealing (Continual QAT).

    Both reach the same `steps` of training.  Variant A trains ternary
    the entire time.  Variant B trains FP for `warmup_frac * steps`
    then switches to ternary.  We compare final ternary-mode quality.

    Important: variant B's final ternary phase has FEWER ternary-update
    steps than variant A — this is the honest comparison the thesis
    implies (annealing makes the ternary regime easier even with less
    ternary training).
    """
    out = {}
    # Cold-start ternary (the standard baseline)
    set_global_seed(seed)
    cold = _build_model(dataset.vocab_size, estimator="ste", hgf=False)
    out["Ternary cold-start"] = train_and_eval(
        cold, dataset, label="Ternary cold-start", device=device,
        steps=steps, batch_size=32, seq_len=64, lr=3e-4,
        notes="STE estimator the entire run",
    )
    # Continual QAT: FP first, then anneal to ternary
    set_global_seed(seed)
    qat = _build_model(dataset.vocab_size, estimator="fp_baseline", hgf=False)
    fp_steps = int(steps * warmup_frac)
    ter_steps = steps - fp_steps
    out["FP warmup"] = train_and_eval(
        qat, dataset, label="FP warmup", device=device,
        steps=fp_steps, batch_size=32, seq_len=64, lr=3e-4,
        notes=f"FP for {fp_steps} steps (warmup phase of Continual QAT)",
    )
    # Switch estimator to STE for the second phase — same weights,
    # same optimizer state lost (we recreate the call so AdamW state
    # restarts; in production NSOS the trainer would preserve it, but
    # for the harness this isolates the precision switch effect cleanly).
    for blk in qat.blocks:
        for layer in [blk.q_proj, blk.k_proj, blk.v_proj, blk.o_proj, blk.ff1, blk.ff2]:
            layer.estimator = "ste"
    out["Ternary after FP warmup"] = train_and_eval(
        qat, dataset, label="Ternary after FP warmup", device=device,
        steps=ter_steps, batch_size=32, seq_len=64, lr=3e-4,
        notes=f"STE for last {ter_steps} steps (the actual ternary phase)",
    )
    return out


def test_denoising_dequant(*, device: torch.device, dataset, steps: int = 1500,
                            seed: int = 42) -> dict:
    """Classical STE vs Denoising Dequantization Transform.

    Same backbone, same seed, same data.  DDT adds a Ridge penalty to
    the backward to pull weights toward their nearest quantization
    grid point, dampening the STE oscillations.
    """
    out = {}
    for label, est in [("STE", "ste"), ("Denoising-Dequant", "denoising_dequant")]:
        set_global_seed(seed)
        model = _build_model(dataset.vocab_size, estimator=est, hgf=False)
        res = train_and_eval(
            model, dataset, label=label, device=device,
            steps=steps, batch_size=32, seq_len=64, lr=3e-4,
            notes=f"estimator={est}",
        )
        out[label] = res
    return out


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    "decoupled_ste_vs_ste": test_decoupled_ste_vs_ste,
    "hgf":                  test_hgf,
    "continual_qat":        test_continual_qat,
    "denoising_dequant":    test_denoising_dequant,
}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Quantization estimator tests.")
    p.add_argument("--test", choices=list(ALL_TESTS), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--out", type=str, default=None,
                   help="If set, write results JSON here.")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quantization] device={device.type} test={args.test} seed={args.seed} steps={args.steps}")

    dataset = load_wikitext_chars(seed=args.seed)
    print(f"[quantization] dataset: vocab={dataset.vocab_size}, "
          f"train={dataset.train_tensor.numel()} chars, eval={dataset.eval_tensor.numel()} chars")

    fn = ALL_TESTS[args.test]
    results = fn(device=device, dataset=dataset, steps=args.steps, seed=args.seed)

    # Print summary
    print(f"\n=== test_{args.test} ===")
    print(f"{'Variant':<35} {'TrainLoss':>10} {'EvalPPL':>10} {'Spikes':>8} {'WallS':>8} {'Params':>10}")
    for label, res in results.items():
        print(f"{res.label:<35} {res.final_train_loss:>10.4f} {res.eval_ppl:>10.2f} "
              f"{res.loss_spikes:>8d} {res.wall_time_s:>8.1f} {res.total_params:>10d}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        print(f"[quantization] wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
