from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from nsos_curriculum_lib import SPECIAL_TOKENS, curriculum_texts_for_phase
from train_curriculum import (
    EVAL_RUNTIME_OPTIONS,
    build_model_config,
    build_supervised_tokens,
    detect_build_dir,
    ensure_bundle,
    evaluate_masked_supervised,
    evaluate_phase,
    greedy_generate,
    load_holdout_jsonl,
    load_nsos,
    model_config_to_dict,
    resolve_profile,
    set_model_training_mode,
)


def timed(label: str, fn: Callable[[], Any], repeat: int, warmup: int) -> Dict[str, Any]:
    for _ in range(max(warmup, 0)):
        fn()
    samples: List[float] = []
    for _ in range(max(repeat, 1)):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return {
        "label": label,
        "repeat": len(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "mean_s": statistics.fmean(samples),
        "median_s": statistics.median(samples),
        "samples_s": samples,
    }


def build_probe_rows(args: argparse.Namespace) -> List[Dict]:
    if args.holdout_file is not None:
        rows = load_holdout_jsonl(args.holdout_file)
    else:
        rows = curriculum_texts_for_phase(args.bundle_dir, args.phase, "eval")
    return [row for row in rows if row.get("prompt") and row.get("answer")][: max(args.rows, 1)]


def make_model(nsos, profile: Dict, tokenizer, device):
    config = build_model_config(nsos, profile, tokenizer.vocab_size, device)
    model = nsos.JambaModel(config, device)
    model.to(device)
    set_model_training_mode(model, False)
    return model, config


def configure_audit(nsos, model, args: argparse.Namespace):
    collector = nsos.LayerAuditCollector()
    collector.begin_run("inference_perf_probe")
    collector.set_storage_policy(
        bool(args.audit_summary_only),
        max(int(args.audit_record_sample_rate), 1),
        max(int(args.audit_max_records_per_phase), 0),
        bool(args.audit_store_token_contexts),
    )
    collector.set_enabled(True)
    model.set_audit_collector(collector)
    return collector


def run_suite(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
              seq_len: int, args: argparse.Namespace, audit_collector=None) -> Dict[str, Any]:
    prompt_row = rows[0]
    prompt = f"<|task:{prompt_row.get('kind', 'probe')}|>\nPrompt:\n{prompt_row['prompt']}\nAnswer:\n"
    prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, prompt_row, eos_token_id)
    forward_inputs = list(prompt_tokens)
    if len(answer_tokens) > 1:
        forward_inputs.extend(answer_tokens[:-1])

    def set_phase(name: str) -> None:
        if audit_collector is not None:
            audit_collector.set_phase(name)

    def forward_once():
        set_phase("forward_ids")
        model.reset_session()
        return model.forward_ids(forward_inputs, None)

    def generate_once():
        set_phase("greedy_generate")
        return greedy_generate(
            nsos,
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            eos_token_id,
            logger=None,
            label="perf:generate",
        )

    def masked_once():
        set_phase("masked_supervised")
        return evaluate_masked_supervised(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            batch_size=args.masked_batch_size,
        )

    def phase_once():
        set_phase("evaluate_phase")
        return evaluate_phase(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            seq_len,
            args.eval_mode,
            args.exact_samples,
            logger=None,
            label="perf:phase",
        )

    return {
        "forward_ids": timed("forward_ids", forward_once, args.repeat, args.warmup),
        "greedy_generate": timed("greedy_generate", generate_once, args.repeat, args.warmup),
        "evaluate_masked_supervised": timed(
            "evaluate_masked_supervised",
            masked_once,
            args.repeat,
            args.warmup,
        ),
        "evaluate_phase": timed("evaluate_phase", phase_once, args.repeat, args.warmup),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure NSOS inference and evaluation hotspots.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_bundle",
    )
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rebuild-curriculum", action="store_true")
    parser.add_argument("--phase", default="phase1_algorithms")
    parser.add_argument("--holdout-file", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--eval-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--exact-samples", type=int, default=0)
    parser.add_argument("--fast-exact-samples", type=int, default=0)
    parser.add_argument("--generation-probe-samples", type=int, default=0)
    parser.add_argument("--masked-batch-size", type=int, default=4)
    parser.add_argument("--text-loss-max-windows", type=int, default=4)
    parser.add_argument("--with-audit", action="store_true")
    parser.add_argument("--audit-summary-only", action="store_true")
    parser.add_argument("--audit-record-sample-rate", type=int, default=8)
    parser.add_argument("--audit-max-records-per-phase", type=int, default=64)
    parser.add_argument(
        "--audit-store-token-contexts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "inference_perf_probe.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_profile, profile = resolve_profile(args.profile)
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)
    tokenizer_path = ensure_bundle(
        args.repo_root,
        args.bundle_dir,
        args.seed,
        int(profile["target_vocab"]),
        args.rebuild_curriculum,
        profile.get("phase_sizes"),
    )
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(tokenizer_path))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    rows = build_probe_rows(args)
    if not rows:
        raise RuntimeError("No probe rows available; provide --holdout-file or rebuild the curriculum bundle.")

    EVAL_RUNTIME_OPTIONS.update({
        "masked_batch_size": max(int(args.masked_batch_size), 1),
        "generation_probe_samples": max(int(args.generation_probe_samples), 0),
        "fast_exact_samples": max(int(args.fast_exact_samples), 0),
        "text_loss_max_windows": max(int(args.text_loss_max_windows), 0),
    })

    model, config = make_model(nsos, profile, tokenizer, device)
    report: Dict[str, Any] = {
        "profile": canonical_profile,
        "requested_profile": args.profile,
        "device": args.device,
        "build_dir": str(build_dir),
        "rows": len(rows),
        "model_config": model_config_to_dict(config),
        "eval_options": dict(EVAL_RUNTIME_OPTIONS),
        "without_audit": run_suite(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            int(profile["seq_len"]),
            args,
        ),
    }

    if args.with_audit:
        audit_model, _ = make_model(nsos, profile, tokenizer, device)
        collector = configure_audit(nsos, audit_model, args)
        report["with_audit"] = run_suite(
            nsos,
            audit_model,
            tokenizer,
            rows,
            eos_token_id,
            int(profile["seq_len"]),
            args,
            audit_collector=collector,
        )
        report["audit"] = {
            "summary_only": collector.summary_only(),
            "record_sample_rate": collector.record_sample_rate(),
            "max_records_per_phase": collector.max_records_per_phase(),
            "store_token_contexts": collector.store_token_contexts(),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
