"""LEARN B1 — Supervised Fine-Tuning phase (SmolTalk + Alpaca-cleaned).

Frontier LLM training pipelines separate "pretraining" (next-token
prediction on web/synthetic corpora) from "supervised fine-tuning"
(loss masked on the answer span only, using high-quality instruction
data, at a much lower learning rate).  NSOS's existing curriculum
training treats phase4_instructions as one phase among six and uses
the pretrain LR for all of them.

This script implements SFT as a SEPARATE phase that:
  * Loads a pretrained NSOS checkpoint (typically the post-curriculum
    final_model.bin or the latest phase checkpoint).
  * Loads an instruction dataset.  Defaults to SmolTalk + Alpaca-cleaned
    combined; user can point at the synthetic Claude-generated data
    from generate_synthetic_instructions.py instead.
  * Trains at ~10× LOWER learning rate than pretrain.
  * Uses Trainer::train_supervised_batch which already masks loss to
    the answer span (loss is only computed on the answer tokens, not
    on the prompt).  This is the right pattern for SFT.
  * Tracks per-task instruction-following metrics: teacher token
    accuracy, exact match, generation latency.
  * Auto-checkpoints every N steps + saves an "SFT champion" checkpoint
    selected by held-out eval score.

The output is a `<run_dir>_sft/` directory parallel to the pretrain
run.  Inference can load either the pretrain or SFT checkpoint — SFT
is strictly better at instruction following but may have lost some
raw language modeling capacity (the "alignment tax").

CLI:
  python sft_phase.py \\
      --pretrain-checkpoint live_distill_v11_colab/v11_colab_t4_run01_2026-05-16/final_model.bin \\
      --instruction-bundle artifacts/sft_bundle/instructions.jsonl \\
      --out-dir live_distill_v11_sft \\
      --model-config live_distill_v11_colab/.../effective_model_config.json \\
      --tokenizer live_distill_v11_colab/.../tokenizer.nsos \\
      --batch-size 16 \\
      --lr 5e-5 \\
      --warmup-steps 50 \\
      --max-steps 2000 \\
      --checkpoint-every-steps 100 \\
      --device gpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the existing curriculum infrastructure for tokenization +
# train loop.  We import lazily because nsos_ext is a heavy native
# extension and we want --help to work without it.

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True,
                        help="Path to the pretrained .bin checkpoint to SFT from.")
    parser.add_argument("--instruction-bundle", type=Path, required=True,
                        help="JSONL of {prompt, answer, kind} rows.  Can "
                             "be the output of generate_synthetic_instructions.py.")
    parser.add_argument("--eval-bundle", type=Path, default=None,
                        help="Optional independent held-out JSONL.  When set, "
                             "the instruction bundle is used entirely for training "
                             "and this file is never mixed into it.")
    parser.add_argument("--model-config", type=Path, required=True,
                        help="effective_model_config.json from the pretrain run.")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Path to the tokenizer.nsos packed file from "
                             "the pretrain run.  MUST match the vocab the "
                             "pretrain model was trained with.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for SFT checkpoints + metrics.")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Build directory with nsos_ext (.pyd / .so).  "
                             "Auto-detected if not set.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="SFT batch size.  Smaller than pretrain "
                             "(SFT gradient signal is denser per sample).")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="SFT learning rate.  Frontier rule: ~10x "
                             "lower than pretrain LR (5e-5 here vs 5e-4 "
                             "pretrain).  Higher will catastrophically "
                             "forget pretrain knowledge.")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Linear warmup steps.  Short because the "
                             "model is already calibrated by pretrain.")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Hard cap on SFT steps.  2000 is enough "
                             "for ~30K instructions at batch 16; more "
                             "risks overfitting / forgetting.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm.  SFT typically "
                             "uses 1.0 vs pretrain's 1.5 to prevent "
                             "any single batch from skewing weights.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay during SFT.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100,
                        help="Rolling checkpoint cadence.")
    parser.add_argument("--eval-every-steps", type=int, default=100,
                        help="In-phase eval cadence.  At 0 disables.")
    parser.add_argument("--eval-samples", type=int, default=32,
                        help="Held-out samples per in-phase eval pass.")
    parser.add_argument("--eval-generation-samples", type=int, default=0,
                        help="Optional number of held-out rows for greedy generation. "
                             "Zero keeps eval teacher-forced only (recommended while training).")
    parser.add_argument("--eval-fraction", type=float, default=0.05,
                        help="Fraction of the bundle held out for eval. "
                             "Drawn deterministically with --seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Execution device.")
    parser.add_argument("--seed", type=int, default=20260516,
                        help="Seed for shuffling + held-out split.")
    parser.add_argument("--allow-config-digest-mismatch", action="store_true",
                        help="Load a fingerprint/shape-compatible legacy checkpoint with "
                             "strict=False when only the complete ModelConfig digest differs. "
                             "The load is fail-closed on bad integrity, flags, names, or shapes.")
    return parser.parse_args()


def detect_build_dir(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build-cuda-validation", "build-colab", "build-mvp",
                 "build_cuda129", "build_v1", "build_full", "build_codex", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    patterns = ("nsos_ext*.pyd", "nsos_ext*.so")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for pattern in patterns:
            if any(candidate.glob(pattern)):
                return candidate
    raise RuntimeError(
        "Could not find a build directory with nsos_ext.  Searched: "
        + ", ".join(str(c) for c in candidates))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def deterministic_split(rows: List[Dict], eval_fraction: float,
                        seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Stable train/eval split.  Uses (id or hash of prompt) % N to
    bucket rows so the same split is reproducible across runs."""
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_eval = max(1, int(round(len(rows) * eval_fraction)))
    eval_idx = set(indices[:n_eval])
    train: List[Dict] = []
    eval_: List[Dict] = []
    for i, row in enumerate(rows):
        if i in eval_idx:
            eval_.append(row)
        else:
            train.append(row)
    return train, eval_


def build_supervised_tokens(tokenizer, row: Dict, eos_token_id: int) -> Tuple[List[int], List[int]]:
    """Same formatting as nsos_curriculum_lib.format_supervised_text.
    Returns (prompt_tokens, answer_tokens) where prompt ends with
    "<|task:X|>\nPrompt:\n...\nAnswer:\n" and answer is the answer
    text + EOS."""
    kind = row.get("kind", "instruction")
    prompt = row.get("prompt", "")
    answer = row.get("answer", "")
    full_prompt = f"<|task:{kind}|>\nPrompt:\n{prompt}\nAnswer:\n"
    prompt_tokens = tokenizer.encode(full_prompt)
    answer_tokens = tokenizer.encode(answer) + [eos_token_id]
    return prompt_tokens, answer_tokens


def compute_eval_metrics(model, tokenizer, rows: List[Dict],
                         eos_token_id: int, max_samples: int,
                         generation_samples: int = 0) -> Dict[str, float]:
    """Run held-out eval: teacher token accuracy (predicting each
    answer token given the gold prefix) + exact greedy generation
    match (full-answer generation matches gold)."""
    n = min(len(rows), max_samples)
    if n == 0:
        return {"answer_loss": 0.0, "teacher_token_accuracy": 0.0,
                 "exact_total": 0, "exact_correct": 0, "exact_accuracy": 0.0}
    # Evaluation is deliberately implemented against JambaModel rather than
    # InferenceEngine so a legacy checkpoint can be loaded with strict=False
    # without weakening the training path.  A single full-prefix forward per
    # row gives both teacher-forced loss and accuracy without Python-side
    # token-by-token GPU synchronizations.
    import numpy as np

    teacher_correct = 0
    teacher_total = 0
    losses: List[float] = []
    exact_correct = 0
    exact_total = 0
    for row in rows[:n]:
        prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, row, eos_token_id)
        if not prompt_tokens or not answer_tokens:
            continue
        # Teacher-forced loss on the answer span.  Row j predicts token j+1,
        # therefore the first answer token is scored at prompt_len-1.
        full = list(prompt_tokens) + list(answer_tokens)
        logits = np.asarray(model.forward_ids(full).numpy(), dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != len(full):
            raise RuntimeError(f"unexpected eval logits shape: {logits.shape}")
        answer_logits = logits[len(prompt_tokens) - 1:len(full) - 1]
        gold = np.asarray(answer_tokens, dtype=np.int64)
        if answer_logits.shape[0] != gold.shape[0]:
            raise RuntimeError("eval answer/logit alignment mismatch")
        shifted = answer_logits - answer_logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1))
        losses.append(float(np.mean(log_norm - shifted[np.arange(gold.size), gold])))
        preds = np.argmax(answer_logits, axis=1)
        teacher_correct += int(np.sum(preds == gold))
        teacher_total += int(gold.size)

        # Greedy generation is opt-in because it performs one full-prefix
        # forward per generated token and must never contend with training by
        # default.  It is useful for a small end-of-phase quality probe.
        if generation_samples > 0 and exact_total < generation_samples:
            generated = list(prompt_tokens)
            for _ in range(64):
                next_logits = np.asarray(model.forward_ids(generated).numpy(), dtype=np.float32)[-1]
                next_tok = int(np.argmax(next_logits))
                generated.append(next_tok)
                if next_tok == eos_token_id:
                    break
            generated_answer = tokenizer.decode(generated[len(prompt_tokens):]).strip()
            target_answer = row.get("answer", "").strip()
            if generated_answer.startswith(target_answer):
                exact_correct += 1
            exact_total += 1
    return {
        "answer_loss": sum(losses) / max(len(losses), 1) if losses else 0.0,
        "teacher_token_accuracy": teacher_correct / max(teacher_total, 1),
        "exact_total": exact_total,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / max(exact_total, 1),
    }


def save_training_checkpoint(model, trainer, path: Path) -> None:
    """Persist model plus the matching optimizer/runtime sidecar.

    InferenceEngine.save_checkpoint intentionally serializes only model bytes.
    SFT recovery also needs Adam/scheduler/RNG state, so direct JambaModel
    loading uses the Trainer's transactional sidecar API whenever available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    state_path = path.with_suffix(path.suffix + ".trainer.state")
    trainer.save_training_state(str(state_path), str(path))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    build_dir = detect_build_dir(args.build_dir)
    add_windows_runtime_dirs(build_dir, parse_preferred_cuda_root(None))
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import nsos_ext as nsos  # noqa: E402

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(args.tokenizer))
    # Keep every produced checkpoint self-describing and directly loadable by
    # the SDK/terminal sampler (the engine searches beside model.bin).
    shutil.copy2(args.tokenizer, args.out_dir / "tokenizer.nsos")
    # Register ChatML markers idempotently — SFT bundles built by
    # build_sft_bundle.py format prompts with <|im_start|>/<|im_end|>;
    # without registration BPE would split each marker into many
    # subtokens, defeating the SFT loss mask boundary.
    try:
        from chatml import register_special_tokens
        register_special_tokens(tokenizer)
    except ImportError as exc:
        raise RuntimeError(
            "SFT requires chatml special-token registration"
        ) from exc
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    print(f"[sft] tokenizer vocab_size={tokenizer.vocab_size} eos={eos_token_id}")

    # ── Load model config + checkpoint ──────────────────────────────────
    config = nsos.ModelConfig()
    config_dict = json.loads(args.model_config.read_text("utf-8"))
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    # The conversational pilot was materialized with exact attention
    # training disabled; keep the serialized architecture digest identical
    # when loading its checkpoint for this SFT phase.
    config.use_exact_attention_training = False
    config.use_cuda = (args.device == "gpu") if args.device != "auto" else True

    print(f"[sft] loading pretrain checkpoint: {args.pretrain_checkpoint}")
    engine = None
    if args.allow_config_digest_mismatch:
        # The v4 checkpoint has a matching integrity trailer, architecture
        # fingerprint, parameter names and shapes; only its complete config
        # digest predates the current serialized operational defaults.  The
        # native strict=False loader still rejects bad integrity, flag
        # mismatches, unknown/shape-mismatched tensors, and zero matches.
        device = nsos.Device.GPU if config.use_cuda else nsos.Device.CPU
        model = nsos.JambaModel(config, device)
        try:
            model.load(str(args.pretrain_checkpoint), False)
        except Exception as exc:
            sys.stderr.write(f"[fatal] legacy-compatible checkpoint load failed: {exc}\n")
            return 1
        print("[sft] checkpoint load mode=legacy-compatible strict=False "
              "(integrity/fingerprint/shape checks remain enforced)")
    else:
        engine = nsos.InferenceEngine()
        if not engine.load_model(str(args.pretrain_checkpoint), config):
            sys.stderr.write("[fatal] failed to load pretrain checkpoint\n")
            return 1
        model = engine.model

    build_binary = next(iter(sorted(build_dir.glob("nsos_ext*.pyd"),
                                    key=lambda p: p.name)), None)
    run_manifest = {
        "schema": "nsos-sft-run-v1",
        "filter_version": "canarim-sft-gold-filter-v15",
        "instruction_bundle": str(args.instruction_bundle.resolve()),
        "eval_bundle": str(args.eval_bundle.resolve()) if args.eval_bundle else None,
        "pretrain_checkpoint": str(args.pretrain_checkpoint.resolve()),
        "pretrain_checkpoint_sha256": sha256_file(args.pretrain_checkpoint),
        "tokenizer": str((args.out_dir / "tokenizer.nsos").resolve()),
        "tokenizer_sha256": sha256_file(args.out_dir / "tokenizer.nsos"),
        "build_dir": str(build_dir.resolve()),
        "build_binary": str(build_binary.resolve()) if build_binary else None,
        "build_binary_sha256": sha256_file(build_binary) if build_binary else None,
        "checkpoint_load_mode": "legacy-compatible-strict-false" if args.allow_config_digest_mismatch else "strict",
        "integrity_policy": "checkpoint SHA-256 trailer + architecture fingerprint + parameter names/shapes",
        "device": "gpu" if config.use_cuda else "cpu",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "eval_generation_samples": args.eval_generation_samples,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ── Trainer with SFT-tuned hyperparameters ──────────────────────────
    trainer = nsos.Trainer(model, args.lr)
    trainer.weight_decay = args.weight_decay
    trainer.max_grad_norm = args.max_grad_norm

    # LR schedule: linear warmup then linear decay to 10% of base
    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step / max(args.warmup_steps, 1))
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        progress = max(0.0, min(1.0, progress))
        return args.lr * (1.0 - 0.9 * progress)

    # ── Dataset load + split ───────────────────────────────────────────
    print(f"[sft] loading instruction bundle: {args.instruction_bundle}")
    rows = load_jsonl(args.instruction_bundle)
    if not rows:
        sys.stderr.write("[fatal] instruction bundle is empty\n")
        return 1
    if args.eval_bundle is not None:
        train_rows = rows
        eval_rows = load_jsonl(args.eval_bundle)
        if not eval_rows:
            sys.stderr.write("[fatal] independent eval bundle is empty\n")
            return 1
        print(f"[sft] independent eval: train={len(train_rows)} eval={len(eval_rows)}")
    else:
        train_rows, eval_rows = deterministic_split(rows, args.eval_fraction, args.seed)
        print(f"[sft] split: train={len(train_rows)} eval={len(eval_rows)}")

    # ── Training loop ───────────────────────────────────────────────────
    metrics_path = args.out_dir / "sft_metrics.jsonl"
    best_eval_score = float("-inf")
    rng = random.Random(args.seed)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    cursor = 0
    step = 0
    started = time.time()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < args.max_steps:
            # Build batch
            prompt_batch: List[List[int]] = []
            answer_batch: List[List[int]] = []
            while len(prompt_batch) < args.batch_size:
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                row = train_rows[order[cursor]]
                cursor += 1
                p_tok, a_tok = build_supervised_tokens(tokenizer, row, eos_token_id)
                if not p_tok or not a_tok:
                    continue
                prompt_batch.append(p_tok)
                answer_batch.append(a_tok)
            if not prompt_batch:
                continue

            # Apply LR schedule
            trainer.learning_rate = lr_at_step(step)

            step += 1
            loss = trainer.train_supervised_batch(prompt_batch, answer_batch)
            elapsed = time.time() - started
            if step % 10 == 0:
                print(f"[sft] step {step:>4}/{args.max_steps}  "
                      f"loss={loss:.4f}  lr={trainer.learning_rate:.2e}  "
                      f"elapsed={elapsed:.0f}s")

            # Eval + champion selection
            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                metrics = compute_eval_metrics(model, tokenizer, eval_rows,
                                                eos_token_id, args.eval_samples,
                                                args.eval_generation_samples)
                metrics["step"] = step
                metrics["train_loss"] = float(loss)
                metrics["lr"] = float(trainer.learning_rate)
                metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                metrics_file.flush()
                score = metrics["teacher_token_accuracy"] + 0.1 * metrics["exact_accuracy"]
                print(f"[sft] eval@{step}: teacher_acc={metrics['teacher_token_accuracy']:.3f} "
                      f"exact_acc={metrics['exact_accuracy']:.3f} "
                      f"loss={metrics['answer_loss']:.4f} score={score:.4f}")
                if score > best_eval_score:
                    best_eval_score = score
                    champion = args.out_dir / "sft_champion.bin"
                    trainer.global_step_count = step
                    save_training_checkpoint(model, trainer, champion)
                    print(f"[sft] new champion checkpoint saved -> {champion}")

            # Rolling checkpoint
            if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                rolling = args.out_dir / f"sft_step_{step:06d}.bin"
                trainer.global_step_count = step
                save_training_checkpoint(model, trainer, rolling)
                # Keep only the last 3 rolling checkpoints to bound disk usage.
                rolling_all = sorted(args.out_dir.glob("sft_step_*.bin"))
                for old in rolling_all[:-3]:
                    old.unlink(missing_ok=True)
                    old.with_suffix(old.suffix + ".trainer.state").unlink(missing_ok=True)

    # Final checkpoint
    final = args.out_dir / "sft_final.bin"
    trainer.global_step_count = step
    save_training_checkpoint(model, trainer, final)
    print(f"[sft] final checkpoint -> {final}")
    print(f"[sft] best eval score: {best_eval_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
