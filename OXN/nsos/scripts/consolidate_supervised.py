from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
from nsos_curriculum_lib import curriculum_texts_for_phase


DEFAULT_PHASES = [
    "phase1_algorithms",
    "phase2_structured",
    "phase4_instructions",
    "phase5_verifier",
    "phase6_memory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast supervised consolidation with quick probes.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--out-run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.25)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--min-lr-scale", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--phases", nargs="*", default=DEFAULT_PHASES)
    return parser.parse_args()


def load_nsos(build_dir: Path):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    import nsos_ext as nsos  # type: ignore

    return nsos


def load_run_metadata(run_dir: Path) -> Dict:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def safe_decode(tokenizer, token_ids: Iterable[int]) -> str:
    ids = list(token_ids)
    try:
        return tokenizer.decode(ids)
    except Exception:
        return "".join(chr(token % 256) for token in ids)


def argmax_token(nsos, logits) -> int:
    host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
    return int(host_logits.numpy()[-1].argmax())


def generate_tokens(nsos, model, prompt_ids: List[int], decode_steps: int, eos_token_id: int) -> List[int]:
    generated: List[int] = []
    model.reset_session()
    streaming = model.supports_streaming_inference()
    if streaming:
        model.set_streaming_inference(True)
        logits = None
        for token in prompt_ids:
            logits = model.forward_ids([token], None)
        for _ in range(decode_steps):
            next_token = argmax_token(nsos, logits)
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            logits = model.forward_ids([next_token], None)
        model.set_streaming_inference(False)
        return generated

    for _ in range(decode_steps):
        model.reset_session()
        logits = model.forward_ids(prompt_ids + generated, None)
        next_token = argmax_token(nsos, logits)
        if next_token == eos_token_id:
            break
        generated.append(next_token)
    return generated


def evaluate_probe(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int) -> Dict[str, float]:
    total = 0
    correct = 0
    teacher_total = 0
    teacher_correct = 0
    for row in rows:
        prompt = f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(row["answer"])
        if not prompt_ids or not answer_ids:
            continue

        generated = generate_tokens(nsos, model, prompt_ids, max(len(answer_ids) + 4, 8), eos_token_id)
        prediction = safe_decode(tokenizer, generated).replace("<|endoftext|>", "").strip()
        if "\n" in prediction:
            prediction = prediction.split("\n", 1)[0].strip()
        if prediction == row["answer"].strip():
            correct += 1
        total += 1

        model.reset_session()
        if model.supports_streaming_inference():
            model.set_streaming_inference(True)
            logits = None
            for token in prompt_ids:
                logits = model.forward_ids([token], None)
            for token in answer_ids:
                teacher_total += 1
                if argmax_token(nsos, logits) == token:
                    teacher_correct += 1
                logits = model.forward_ids([token], None)
            model.set_streaming_inference(False)
        else:
            prefix = list(prompt_ids)
            for token in answer_ids:
                logits = model.forward_ids(prefix, None)
                teacher_total += 1
                if argmax_token(nsos, logits) == token:
                    teacher_correct += 1
                prefix.append(token)

    return {
        "exact_total": total,
        "exact_correct": correct,
        "exact_accuracy": (correct / total) if total else 0.0,
        "teacher_total": teacher_total,
        "teacher_correct": teacher_correct,
        "teacher_accuracy": (teacher_correct / teacher_total) if teacher_total else 0.0,
    }


def main() -> int:
    args = parse_args()
    args.out_run_dir.mkdir(parents=True, exist_ok=True)

    nsos = load_nsos(args.build_dir)
    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU

    meta = load_run_metadata(args.source_run_dir)
    model = nsos.JambaModel(
        int(meta["model"]["layers"]),
        int(meta["model"]["d_model"]),
        int(meta["model"]["vocab_size"]),
        device,
    )
    model.to(device)
    model.load(str(args.source_run_dir / "final_model.bin"))

    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(args.source_run_dir / "tokenizer.nsos"))
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    trainer = nsos.Trainer(model, args.learning_rate)
    trainer.weight_decay = args.weight_decay
    trainer.max_grad_norm = args.max_grad_norm
    trainer.warmup_steps = args.warmup_steps
    trainer.min_learning_rate_scale = args.min_lr_scale

    train_rows: List[Dict] = []
    probe_rows: Dict[str, List[Dict]] = {}
    for phase in args.phases:
        train_rows.extend(curriculum_texts_for_phase(args.bundle_dir, phase, "train"))
        probe_rows[phase] = curriculum_texts_for_phase(args.bundle_dir, phase, "eval")[: args.eval_samples]

    rng = random.Random(args.seed)
    probes: List[Dict] = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        row = train_rows[rng.randrange(len(train_rows))]
        prompt_tokens = tokenizer.encode(f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n")
        answer_tokens = tokenizer.encode(row["answer"]) + [eos_token_id]
        loss = trainer.train_supervised(prompt_tokens, answer_tokens)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            snapshot = {"step": step, "loss": float(loss), "phases": {}}
            for phase in args.phases:
                snapshot["phases"][phase] = evaluate_probe(
                    nsos, model, tokenizer, probe_rows[phase], eos_token_id
                )
            probes.append(snapshot)
            print(f"[probe] step={step} loss={loss:.4f}")
            for phase in args.phases:
                metrics = snapshot["phases"][phase]
                print(
                    f"  {phase}: exact={metrics['exact_correct']}/{metrics['exact_total']} "
                    f"teacher={metrics['teacher_correct']}/{metrics['teacher_total']}"
                )

    elapsed = time.perf_counter() - started
    model.save(str(args.out_run_dir / "final_model.bin"))
    model.save_edge_linear_pack(str(args.out_run_dir / "final_edge_linear.nsos"))
    shutil.copy2(args.source_run_dir / "tokenizer.nsos", args.out_run_dir / "tokenizer.nsos")

    report = {
        "source_run_dir": str(args.source_run_dir),
        "out_run_dir": str(args.out_run_dir),
        "bundle_dir": str(args.bundle_dir),
        "device": args.device,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": args.warmup_steps,
        "min_lr_scale": args.min_lr_scale,
        "phases": args.phases,
        "elapsed_s": elapsed,
        "probes": probes,
    }
    (args.out_run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] saved consolidated run to {args.out_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
