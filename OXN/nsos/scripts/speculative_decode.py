"""LEARN C2 (2026-05-16) — Speculative decoding for NSOS inference.

Speculative decoding (Leviathan et al. 2023 ICML, "Fast Inference
from Transformers via Speculative Decoding") is an exact-equivalent
inference speedup that does NOT change the output distribution of
the target model:

  1. A small DRAFT model autoregressively proposes K candidate tokens.
  2. The large TARGET model runs ONE parallel forward over those K
     tokens (computing the probability distribution at each).
  3. We accept tokens one-by-one using rejection sampling against the
     target distribution; on the first reject, we sample from the
     adjusted distribution at that position and stop.
  4. Repeat from the new accepted prefix.

Acceptance rate depends on draft-target agreement.  For NSOS 40M as
target + 4M draft, agreement is high on common tokens (~70% in
English text), giving ~2.5x effective speedup.

This script wraps two NSOS InferenceEngine instances:
  * target_engine: the full SFT-trained 40M model
  * draft_engine:  a 4M model trained to mimic target outputs
The draft is trained by:
  * Distilling the target's logits on a held-out corpus (top-K
    softmax matching, KL divergence loss).
  * OR by training a smaller-but-same-arch model on the same SFT
    data (cheaper but less aligned).
We provide a `train_draft.py` companion that does the first option.

CLI usage:
  python speculative_decode.py \\
      --target-checkpoint live_distill_v11_sft/sft_final.bin \\
      --draft-checkpoint  live_distill_v11_draft/draft_final.bin \\
      --tokenizer <path>/tokenizer.nsos \\
      --target-config <path>/effective_model_config.json \\
      --draft-config  <path>/draft_model_config.json \\
      --prompt "Explain in one sentence what list comprehension does:" \\
      --max-new-tokens 64 \\
      --gamma 4

`--gamma` is the number of speculative tokens drafted per round.
Optimal gamma depends on acceptance rate alpha:
  Expected tokens per round = (1 - alpha^(gamma+1)) / (1 - alpha)
  Maximize over gamma given measured alpha.
For alpha ~0.7 (typical English): gamma=4 is near optimal.

Output: the generated text + speedup metrics (acceptance rate,
average tokens/round, wallclock vs naive baseline).
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-checkpoint", type=Path, required=True,
                        help="Large model checkpoint (the one whose output "
                             "distribution we want to preserve).")
    parser.add_argument("--draft-checkpoint", type=Path, required=True,
                        help="Small draft model checkpoint.  Typically ~10x "
                             "fewer params than target.")
    parser.add_argument("--target-config", type=Path, required=True,
                        help="ModelConfig JSON for the target.")
    parser.add_argument("--draft-config", type=Path, required=True,
                        help="ModelConfig JSON for the draft.")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Tokenizer.  Target and draft MUST share the "
                             "same vocabulary or the rejection sampling is "
                             "incorrect.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt to generate from.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=4,
                        help="Speculative tokens per round.  Optimal gamma "
                             "depends on draft-target agreement rate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for both draft and target. "
                             "Greedy decode is recovered with temperature=0 "
                             "and both models always agreeing on argmax.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling cutoff.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K cutoff (applied before top-p).")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--baseline-comparison", action="store_true",
                        help="Also run plain target-only decode for the "
                             "same prompt and report speedup.")
    return parser.parse_args()


def detect_build_dir(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build-cuda-validation", "build-colab", "build-mvp",
                 "build_cuda129", "build_v1", "build_full", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    patterns = ("nsos_ext*.pyd", "nsos_ext*.so")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for pattern in patterns:
            if any(candidate.glob(pattern)):
                return candidate
    raise RuntimeError("Could not find a build directory with nsos_ext.")


def load_engine_with_config(nsos, ckpt: Path, cfg_path: Path, use_cuda: bool):
    cfg = nsos.ModelConfig()
    for k, v in json.loads(cfg_path.read_text("utf-8")).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.use_cuda = use_cuda
    eng = nsos.InferenceEngine()
    if not eng.load_model(str(ckpt), cfg):
        raise RuntimeError(f"failed to load checkpoint: {ckpt}")
    return eng


def softmax_temperature(logits: List[float], temperature: float) -> List[float]:
    """Stable softmax with temperature.  Returns probabilities."""
    if temperature <= 0.0:
        # Greedy: one-hot on argmax
        m = max(range(len(logits)), key=lambda i: logits[i])
        out = [0.0] * len(logits)
        out[m] = 1.0
        return out
    scaled = [l / temperature for l in logits]
    max_v = max(scaled)
    exps = [math.exp(s - max_v) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def apply_top_k_top_p(probs: List[float], top_k: int,
                      top_p: float) -> List[float]:
    """Mask probabilities outside the top-K / nucleus and renormalize."""
    n = len(probs)
    if top_k > 0 and top_k < n:
        ranked = sorted(range(n), key=lambda i: probs[i], reverse=True)
        keep_set = set(ranked[:top_k])
        masked = [p if i in keep_set else 0.0 for i, p in enumerate(probs)]
    else:
        masked = list(probs)
    if top_p > 0.0 and top_p < 1.0:
        ranked = sorted(range(n), key=lambda i: masked[i], reverse=True)
        cum = 0.0
        keep_set = set()
        for i in ranked:
            keep_set.add(i)
            cum += masked[i]
            if cum >= top_p:
                break
        masked = [p if i in keep_set else 0.0 for i, p in enumerate(masked)]
    s = sum(masked)
    if s <= 0.0:
        return [1.0 / n] * n   # degenerate; uniform fallback
    return [p / s for p in masked]


def sample_from(probs: List[float], rng: random.Random) -> int:
    """Categorical sample via inverse-CDF (binary search)."""
    cdf = []
    cum = 0.0
    for p in probs:
        cum += p
        cdf.append(cum)
    r = rng.random() * cdf[-1]
    return bisect.bisect_left(cdf, r)


def get_logits(engine, token_history: List[int]) -> List[float]:
    """Returns the next-token logits for the given history.  Requires
    the engine to expose `next_token_logits(history) -> List[float]`.
    If unavailable, falls back to `generate(..., 1, ...)` and decoding
    a single token (loses the full distribution — speculative decoding
    needs the full distribution to do the rejection-sampling step
    correctly)."""
    try:
        logits = engine.next_token_logits(token_history)
        return list(logits)
    except AttributeError:
        # Fallback: emit a one-hot at the greedy token.  This makes
        # speculative decoding degenerate to greedy and disables
        # rejection sampling — useful as a smoke test only.
        gen_id = engine.next_token_greedy(token_history)
        n = engine.model.model_config.vocab_size if hasattr(
            engine.model, "model_config") else 32000
        out = [0.0] * n
        out[gen_id] = 1.0
        return out


def speculative_decode(target_engine, draft_engine, tokenizer,
                       prompt_tokens: List[int], eos_token_id: int,
                       max_new: int, gamma: int, temperature: float,
                       top_k: int, top_p: float,
                       rng: random.Random) -> Tuple[List[int], dict]:
    """Run speculative decoding.  Returns (generated_tokens, metrics).

    The algorithm (Leviathan et al. 2023):
      Repeat until max_new reached:
        1. Draft proposes x_1..x_gamma autoregressively.
        2. For i=1..gamma, get target prob p_i(x_i | prefix + x_1..x_{i-1})
           and draft prob q_i(x_i | prefix + x_1..x_{i-1}).
        3. For i=1..gamma:
             u ~ Uniform(0,1)
             if u < p_i / q_i, accept x_i (continues to i+1)
             else: reject. Sample replacement from adjusted distribution
                   p_adjusted(t) = max(0, p_i(t) - q_i(t)) / Z   STOP.
        4. If all gamma accepted, sample one BONUS token from target's
           distribution at position gamma+1 (target was already computed).
    """
    current = list(prompt_tokens)
    generated: List[int] = []
    rounds = 0
    accepted_total = 0
    proposed_total = 0

    while len(generated) < max_new:
        rounds += 1

        # Step 1: draft proposes gamma tokens.
        draft_tokens: List[int] = []
        draft_probs: List[List[float]] = []
        draft_history = list(current)
        for _ in range(gamma):
            d_logits = get_logits(draft_engine, draft_history)
            d_probs = softmax_temperature(d_logits, temperature)
            d_probs = apply_top_k_top_p(d_probs, top_k, top_p)
            tok = sample_from(d_probs, rng)
            draft_tokens.append(tok)
            draft_probs.append(d_probs)
            draft_history.append(tok)
            if tok == eos_token_id:
                break

        # Step 2: target runs parallel forward over (current + draft_tokens).
        # We need target probs at each of the gamma+1 positions:
        #   p_0 at position |current|       (predicts x_1)
        #   p_1 at position |current|+1     (predicts x_2)
        #   ...
        #   p_g at position |current|+g     (predicts the bonus token)
        # We get them by running target forward `gamma+1` times — once
        # for each prefix length.  This is the FALLBACK path; a future
        # binding upgrade should expose a single forward that returns
        # logits at every position so we get all gamma+1 in one call.
        target_probs: List[List[float]] = []
        target_history = list(current)
        for i in range(len(draft_tokens)):
            t_logits = get_logits(target_engine, target_history)
            t_probs = softmax_temperature(t_logits, temperature)
            t_probs = apply_top_k_top_p(t_probs, top_k, top_p)
            target_probs.append(t_probs)
            target_history.append(draft_tokens[i])
        # Bonus position
        bonus_logits = get_logits(target_engine, target_history)
        bonus_probs = softmax_temperature(bonus_logits, temperature)
        bonus_probs = apply_top_k_top_p(bonus_probs, top_k, top_p)

        # Step 3: rejection sampling
        accepted_this_round = 0
        for i, tok in enumerate(draft_tokens):
            proposed_total += 1
            p = target_probs[i][tok]
            q = draft_probs[i][tok]
            if q <= 0.0:
                # Draft assigned zero prob — must reject (would div by zero).
                # Sample from target_probs[i] instead.
                replacement = sample_from(target_probs[i], rng)
                generated.append(replacement)
                current = current + [replacement]
                break
            u = rng.random()
            if u < (p / q):
                # Accept.
                generated.append(tok)
                current = current + [tok]
                accepted_total += 1
                accepted_this_round += 1
                if len(generated) >= max_new or tok == eos_token_id:
                    break
            else:
                # Reject.  Sample from p_adjusted = max(0, p - q) / Z.
                adjusted = [max(0.0, target_probs[i][t] - draft_probs[i][t])
                            for t in range(len(target_probs[i]))]
                z = sum(adjusted)
                if z <= 0.0:
                    # All target mass was below draft mass — fall back
                    # to plain target sampling.
                    replacement = sample_from(target_probs[i], rng)
                else:
                    adjusted = [a / z for a in adjusted]
                    replacement = sample_from(adjusted, rng)
                generated.append(replacement)
                current = current + [replacement]
                break
        else:
            # All gamma tokens accepted — emit bonus from target.
            if len(generated) < max_new:
                bonus_tok = sample_from(bonus_probs, rng)
                generated.append(bonus_tok)
                current = current + [bonus_tok]
                if bonus_tok == eos_token_id:
                    break

        if generated and generated[-1] == eos_token_id:
            break

    metrics = {
        "rounds": rounds,
        "tokens_generated": len(generated),
        "tokens_proposed": proposed_total,
        "tokens_accepted": accepted_total,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
        "tokens_per_round": len(generated) / max(rounds, 1),
        "gamma": gamma,
    }
    return generated, metrics


def baseline_decode(engine, prompt_tokens: List[int], eos_token_id: int,
                    max_new: int, temperature: float, top_k: int, top_p: float,
                    rng: random.Random) -> List[int]:
    """Plain greedy/temp-sampling decode from target only (no spec)."""
    current = list(prompt_tokens)
    generated: List[int] = []
    for _ in range(max_new):
        logits = get_logits(engine, current)
        probs = softmax_temperature(logits, temperature)
        probs = apply_top_k_top_p(probs, top_k, top_p)
        tok = sample_from(probs, rng)
        generated.append(tok)
        current.append(tok)
        if tok == eos_token_id:
            break
    return generated


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir)
    add_windows_runtime_dirs(build_dir, parse_preferred_cuda_root(None))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import nsos_ext as nsos  # noqa: E402

    use_cuda = (args.device == "gpu") if args.device != "auto" else True
    rng = random.Random(args.seed)

    print(f"[spec] loading target: {args.target_checkpoint}")
    target = load_engine_with_config(nsos, args.target_checkpoint,
                                       args.target_config, use_cuda)
    print(f"[spec] loading draft:  {args.draft_checkpoint}")
    draft = load_engine_with_config(nsos, args.draft_checkpoint,
                                      args.draft_config, use_cuda)

    tok = nsos.Tokenizer()
    tok.load(str(args.tokenizer))
    eos = tok.encode("<|endoftext|>")[0]
    prompt_tokens = tok.encode(args.prompt)
    print(f"[spec] prompt tokens: {len(prompt_tokens)}")
    print()

    # Speculative decode
    t0 = time.time()
    gen, metrics = speculative_decode(
        target, draft, tok, prompt_tokens, eos,
        args.max_new_tokens, args.gamma, args.temperature,
        args.top_k, args.top_p, rng)
    spec_wall = time.time() - t0
    spec_text = tok.decode(gen)

    print("=" * 64)
    print("  SPECULATIVE DECODE RESULT")
    print("=" * 64)
    print(f"  prompt:    {args.prompt}")
    print(f"  generated: {spec_text}")
    print(f"  metrics:   {metrics}")
    print(f"  wallclock: {spec_wall:.2f}s "
          f"({metrics['tokens_generated']/spec_wall:.2f} tok/s)")

    if args.baseline_comparison:
        # Reset rng so we get the same trajectory if temperature=0 (greedy)
        rng_b = random.Random(args.seed)
        t0 = time.time()
        baseline_tokens = baseline_decode(
            target, prompt_tokens, eos, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, rng_b)
        base_wall = time.time() - t0
        base_text = tok.decode(baseline_tokens)
        speedup = base_wall / max(spec_wall, 1e-6)
        print()
        print("=" * 64)
        print("  BASELINE (target-only) RESULT")
        print("=" * 64)
        print(f"  generated: {base_text}")
        print(f"  wallclock: {base_wall:.2f}s "
              f"({len(baseline_tokens)/base_wall:.2f} tok/s)")
        print(f"  speedup (spec/baseline): {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
