from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

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
    parser = argparse.ArgumentParser(description="Fast supervised consolidation with cheap probe metrics.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--out-run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
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


def argmax_token(nsos, logits) -> int:
    host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
    return int(host_logits.numpy()[-1].argmax())


def build_prompt_answer(tokenizer, row: Dict, eos_token_id: int):
    prompt_tokens = tokenizer.encode(
        f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
    )
    answer_tokens = tokenizer.encode(row["answer"]) + [eos_token_id]
    return prompt_tokens, answer_tokens


def masked_answer_metrics(nsos, model, prompt_tokens: List[int], answer_tokens: List[int]) -> Dict[str, float]:
    inputs = list(prompt_tokens)
    if len(answer_tokens) > 1:
        inputs.extend(answer_tokens[:-1])

    model.reset_session()
    logits = model.forward_ids(inputs, None)
    start = len(prompt_tokens) - 1
    end = start + len(answer_tokens)
    answer_logits = logits.slice(0, start, end)
    host_logits = answer_logits.cpu() if answer_logits.device == nsos.Device.GPU else answer_logits
    loss, _ = host_logits.cross_entropy(answer_tokens)

    first_row = host_logits.numpy()[0]
    first_token = int(first_row.argmax())
    return {
        "loss": float(loss),
        "first_token_correct": 1.0 if first_token == answer_tokens[0] else 0.0,
    }


def evaluate_fast_probe(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int) -> Dict[str, float]:
    if not rows:
        return {"avg_loss": 0.0, "first_token_accuracy": 0.0, "samples": 0}
    losses: List[float] = []
    first_hits = 0.0
    for row in rows:
        prompt_tokens, answer_tokens = build_prompt_answer(tokenizer, row, eos_token_id)
        metrics = masked_answer_metrics(nsos, model, prompt_tokens, answer_tokens)
        losses.append(metrics["loss"])
        first_hits += metrics["first_token_correct"]
    return {
        "avg_loss": sum(losses) / len(losses),
        "first_token_accuracy": first_hits / len(rows),
        "samples": len(rows),
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
    snapshots: List[Dict] = []
    started = time.perf_counter()

    def run_snapshot(step: int, loss_value: float) -> None:
        snapshot = {"step": step, "loss": float(loss_value), "phases": {}}
        for phase in args.phases:
            snapshot["phases"][phase] = evaluate_fast_probe(
                nsos, model, tokenizer, probe_rows[phase], eos_token_id
            )
        snapshots.append(snapshot)
        print(f"[fast] step={step} loss={loss_value:.4f}")
        for phase in args.phases:
            metrics = snapshot["phases"][phase]
            print(
                f"  {phase}: first={metrics['first_token_accuracy']:.2f} "
                f"loss={metrics['avg_loss']:.3f} samples={metrics['samples']}"
            )

    initial_row = train_rows[0]
    initial_prompt, initial_answer = build_prompt_answer(tokenizer, initial_row, eos_token_id)
    initial_loss = masked_answer_metrics(nsos, model, initial_prompt, initial_answer)["loss"]
    run_snapshot(0, initial_loss)

    for step in range(1, args.steps + 1):
        row = train_rows[rng.randrange(len(train_rows))]
        prompt_tokens, answer_tokens = build_prompt_answer(tokenizer, row, eos_token_id)
        loss = trainer.train_supervised(prompt_tokens, answer_tokens)
        if step % args.eval_every == 0 or step == args.steps:
            run_snapshot(step, float(loss))

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
        "phases": args.phases,
        "elapsed_s": elapsed,
        "snapshots": snapshots,
    }
    (args.out_run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] saved fast consolidated run to {args.out_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
