"""Run the full test suite (or a subset) and print a summary table.

Usage:
  python run_all.py --quick                # ~30 min on T4, ~3 h on CPU
  python run_all.py --full                 # ~3-4 h on T4 (recommended)
  python run_all.py --only decoupled_ste_vs_ste hgf
  python run_all.py --skip lrcssm_vs_mamba2 cfc_vs_rnn   # if libs unavailable

Results land in `results/<timestamp>/<test>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Eager datasets import before torch — Windows segfault workaround;
# see common/data.py docstring for details.
import datasets  # eager, before torch  # noqa: F401

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load_wikitext_chars


# Registry: (test_name, module_path, kwargs) — kwargs separated for
# --quick vs --full step counts.
TESTS = [
    # (display_name, "module:function", quick_kwargs, full_kwargs, needs_dataset)
    ("decoupled_ste_vs_ste", "tests.quantization:test_decoupled_ste_vs_ste",
     dict(steps=600), dict(steps=2000), True),
    ("hgf", "tests.quantization:test_hgf",
     dict(steps=600), dict(steps=2000), True),
    ("continual_qat", "tests.quantization:test_continual_qat",
     dict(steps=800), dict(steps=2500), True),
    ("denoising_dequant", "tests.quantization:test_denoising_dequant",
     dict(steps=600), dict(steps=2000), True),
    ("lrcssm_vs_mamba2", "tests.sequence_models:test_lrcssm_vs_mamba2",
     dict(steps=400), dict(steps=1500), True),
    ("cfc_vs_rnn", "tests.sequence_models:test_cfc_vs_rnn",
     dict(steps=400), dict(steps=1500), True),
    ("kan_vs_mlp", "tests.kan_variants:test_kan_vs_mlp",
     dict(steps=800), dict(steps=3000), False),
    ("skan_vs_kan", "tests.kan_variants:test_skan_vs_kan",
     dict(steps=800), dict(steps=3000), False),
    ("kantize_latency", "tests.kan_variants:test_kantize_latency",
     dict(steps=300), dict(steps=800), False),
    ("nca_pretrain", "tests.nca_pretrain:test_nca_pretrain",
     dict(steps=600, nca_rollouts=20), dict(steps=2000, nca_rollouts=80), True),
    ("grpo_vs_ppo", "tests.rl_grpo:test_grpo_vs_ppo",
     dict(total_updates=20, episodes_per_update=4),
     dict(total_updates=80, episodes_per_update=8), False),
    ("ab_mcts_vs_uct", "tests.search_mcts:test_ab_mcts_vs_uct",
     dict(n_trees=15, n_simulations=100),
     dict(n_trees=50, n_simulations=400), False),
]


def _resolve_fn(spec: str):
    mod_path, fn_name = spec.split(":")
    mod = __import__(mod_path, fromlist=[fn_name])
    return getattr(mod, fn_name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Theses validation suite orchestrator.")
    p.add_argument("--quick", action="store_true",
                   help="Reduced step counts; ~30 min on T4.")
    p.add_argument("--full", action="store_true",
                   help="Full step counts; ~3-4 h on T4.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these test names.")
    p.add_argument("--skip", nargs="*", default=None,
                   help="Skip these test names.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", type=str, default=None,
                   help="Override results directory (default: results/<timestamp>).")
    args = p.parse_args(argv)

    if not args.quick and not args.full:
        print("Pick --quick or --full.  Quick = ~30 min on T4, Full = ~3-4 h on T4.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"  Theses validation suite — device: {device.type.upper()}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*72}\n")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir or (Path(__file__).parent / "results" / timestamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results -> {out_dir}\n")

    # Load dataset once and share across language tests
    print("  Loading WikiText-2 (char-level)...")
    dataset = load_wikitext_chars(seed=args.seed)
    print(f"  Vocab={dataset.vocab_size}, train={dataset.train_tensor.numel()} chars, "
          f"eval={dataset.eval_tensor.numel()} chars\n")

    summary = []
    for name, spec, qk, fk, needs_ds in TESTS:
        if args.only and name not in args.only:
            continue
        if args.skip and name in args.skip:
            continue
        kw = qk if args.quick else fk

        print(f"  >>> RUN  {name}")
        t0 = time.time()
        try:
            fn = _resolve_fn(spec)
            call_kw = dict(seed=args.seed, **kw)
            if needs_ds:
                call_kw["dataset"] = dataset
            if "device" in fn.__code__.co_varnames:
                call_kw["device"] = device
            res = fn(**call_kw)

            # Serialize
            out_path = out_dir / f"{name}.json"
            try:
                serializable = _serialize(res)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2)
            except Exception as e:
                print(f"    [warn] couldn't serialize {name}: {e}")

            elapsed = time.time() - t0
            print(f"  <<< DONE {name}  ({elapsed:.1f}s)")
            summary.append((name, "ok", elapsed, _short_metric(res)))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  <<< FAIL {name}  ({elapsed:.1f}s): {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append((name, "fail", elapsed, str(e)[:80]))

    # Print final summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'test':<25} {'status':<8} {'time(s)':>10}  metric")
    print(f"  {'-'*25} {'-'*8} {'-'*10}  {'-'*30}")
    for name, status, t, metric in summary:
        print(f"  {name:<25} {status:<8} {t:>10.1f}  {metric}")
    print(f"\n  Full results in: {out_dir}\n")
    return 0


def _serialize(res):
    """Best-effort JSON serialization for whatever shape a test returns."""
    if isinstance(res, dict):
        return {k: _serialize(v) for k, v in res.items()}
    if isinstance(res, (list, tuple)):
        return [_serialize(v) for v in res]
    # RunResult dataclass
    if hasattr(res, "to_dict"):
        return res.to_dict()
    if isinstance(res, (int, float, str, bool)) or res is None:
        return res
    # Fallback
    return repr(res)


def _short_metric(res) -> str:
    """One-line headline for the summary table."""
    if not isinstance(res, dict):
        return repr(res)[:50]
    parts = []
    for k, v in res.items():
        if hasattr(v, "eval_ppl"):
            parts.append(f"{k}: PPL={v.eval_ppl:.1f}")
        elif isinstance(v, dict) and "eval_mse" in v:
            parts.append(f"{k}: MSE={v['eval_mse']:.3f}")
        elif isinstance(v, dict) and "final_avg_return" in v:
            parts.append(f"{k}: ret={v['final_avg_return']:.0f}")
        elif isinstance(v, dict) and "mean_best_leaf" in v:
            parts.append(f"{k}: best={v['mean_best_leaf']:.2f}")
    return " | ".join(parts[:3]) or "<no headline>"


if __name__ == "__main__":
    sys.exit(main())
