from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_curriculum import (
    PROFILES,
    build_supervised_tokens,
    curriculum_texts_for_phase,
    detect_build_dir,
    ensure_bundle,
    load_nsos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce progressive-QAT activation on a short supervised batch."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_bundle_small",
        help="Curriculum bundle directory.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution device.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="small",
        help="Training profile to mirror.",
    )
    parser.add_argument(
        "--phase",
        default="phase6_memory",
        help="Curriculum phase used to build the supervised batches.",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Optional checkpoint to load before reproducing.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of supervised batch steps to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used to ensure the bundle exists.",
    )
    parser.add_argument(
        "--force-qat-active",
        action="store_true",
        help="Force the trainer global step to the QAT threshold before stepping.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)

    tokenizer_path = ensure_bundle(
        args.repo_root,
        args.bundle_dir,
        seed=args.seed,
        target_vocab=profile["target_vocab"],
        rebuild=False,
    )
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(tokenizer_path))
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    model = nsos.JambaModel(profile["layers"], profile["d_model"], tokenizer.vocab_size, device)
    model.to(device)
    if args.resume_model is not None and args.resume_model.exists():
        try:
            model.load(str(args.resume_model), True)
        except TypeError:
            model.load(str(args.resume_model))
        except RuntimeError:
            try:
                model.load(str(args.resume_model), False)
            except TypeError:
                model.load(str(args.resume_model))

    trainer = nsos.Trainer(model, profile["lr"])
    trainer.weight_decay = profile["weight_decay"]
    trainer.max_grad_norm = profile["max_grad_norm"]
    trainer.warmup_steps = profile["warmup_steps"]
    trainer.min_learning_rate_scale = profile["min_lr_scale"]
    trainer.first_token_loss_scale = profile.get("first_token_loss_scale", 2.5)
    trainer.eos_loss_scale = profile.get("eos_loss_scale", 0.35)
    trainer.repetition_unlikelihood_scale = float(
        profile.get("phase_repetition_unlikelihood_scale", {}).get(
            args.phase, profile.get("repetition_unlikelihood_scale", 0.0)
        )
    )
    trainer.eos_token_id = eos_token_id
    trainer.total_training_steps = max(args.steps + 4, 8)

    qat_cfg = profile.get("qat", {})
    scheduler = nsos.TrainPhaseScheduler()
    scheduler.progressive_qat_enabled = True
    scheduler.semantic_warmup_steps = int(qat_cfg.get("semantic_warmup_steps", 0))
    scheduler.qat_start_step = int(qat_cfg.get("qat_start_step", scheduler.semantic_warmup_steps))
    scheduler.quantized_precision_bits = int(qat_cfg.get("quantized_precision_bits", 2))
    scheduler.ternary_regularization = float(qat_cfg.get("ternary_regularization", 0.0))
    trainer.configure_progressive_qat(scheduler)

    if args.force_qat_active:
        trainer.global_step_count = max(int(qat_cfg.get("qat_start_step", 1)), 1)

    rows = [row for row in curriculum_texts_for_phase(args.bundle_dir, args.phase, "train") if row.get("answer")]
    if not rows:
        raise RuntimeError(f"No supervised rows found for phase {args.phase}")

    prompt_batch = []
    answer_batch = []
    answer_lengths = []
    for row in rows[: max(1, profile["batch_size"])]:
        prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, row, eos_token_id)
        if not prompt_tokens or not answer_tokens:
            continue
        prompt_batch.append(prompt_tokens)
        answer_batch.append(answer_tokens)
        answer_lengths.append(len(answer_tokens))

    if not prompt_batch:
        raise RuntimeError("Could not build any supervised prompt/answer pairs")

    report = {
        "profile": args.profile,
        "phase": args.phase,
        "device": args.device,
        "steps_requested": args.steps,
        "force_qat_active": bool(args.force_qat_active),
        "batch_size": len(prompt_batch),
        "answer_lengths": answer_lengths,
        "scheduler": {
            "semantic_warmup_steps": scheduler.semantic_warmup_steps,
            "qat_start_step": scheduler.qat_start_step,
            "quantized_precision_bits": scheduler.quantized_precision_bits,
            "ternary_regularization": scheduler.ternary_regularization,
        },
    }

    try:
        losses = []
        for step in range(args.steps):
            loss = trainer.train_supervised_batch(prompt_batch, answer_batch)
            losses.append(float(loss))
        report["ok"] = True
        report["losses"] = losses
        report["final_global_step"] = int(trainer.global_step_count)
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["ok"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["final_global_step"] = int(getattr(trainer, "global_step_count", -1))

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
