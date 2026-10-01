#!/usr/bin/env python3
"""Run a protocol-identical Mamba-only quality probe across NSOS binaries.

This is intentionally smaller than the product benchmark.  Its purpose is to
locate a quality regression between preserved build directories, including
older extensions that predate runtime telemetry and checkpoint-manifest APIs.
It keeps the model, dataset, sampling, optimizer, and evaluation contract used
by ``benchmark_product_architecture.py`` while recording the exact extension
binary hash loaded for each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import benchmark_product_architecture as product_benchmark


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "artifacts" / "oxta_contabil_amd" / "data",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help="optimizer/scheduler horizon; defaults to --steps",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat every seed in a fresh model to measure HIP nondeterminism",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable ordered reductions, deterministic GPU Mamba, and optimizer",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extension_path(nsos: Any) -> Path:
    path = Path(nsos.__file__).resolve()
    if not path.is_file():
        raise RuntimeError(f"loaded NSOS extension is not a file: {path}")
    return path


def set_if_supported(config: Any, name: str, value: Any) -> None:
    if hasattr(config, name):
        setattr(config, name, value)


def build_mamba_only(
    nsos: Any, vocab_size: int, seed: int, deterministic: bool
):
    nsos.set_seed(seed)
    if hasattr(nsos, "set_deterministic_reductions"):
        nsos.set_deterministic_reductions(deterministic)

    config = nsos.ModelConfig()
    # Fields common to the historical and current Mamba-only benchmark.
    values = {
        "architecture_schema_version": 2,
        "num_layers": product_benchmark.MODEL_LAYERS,
        "d_model": product_benchmark.MODEL_WIDTH,
        "vocab_size": vocab_size,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_context_tokens": 128,
        "default_batch_size": 32,
        "use_cuda": True,
        "dropout": 0.0,
        "use_moe": False,
        "use_kan": False,
        "use_ttt": False,
        "use_chrass": False,
        "use_slender_embedding": False,
        "mamba2_faithful": True,
        "mamba_expand": product_benchmark.MAMBA_EXPAND,
        "mamba_head_dim": 64,
        "mamba_state_expansion": True,
        "mamba_d_state": product_benchmark.MAMBA_STATE,
        "mamba_n_groups": 1,
        "use_gradient_checkpointing": False,
        "use_flash_attn": False,
        "force_mamba_last_layer": False,
        "faithful_attention_linears": True,
        "attention_period": 64,
        "attention_slot": 63,
        "use_exact_attention_training": False,
    }
    for name, value in values.items():
        set_if_supported(config, name, value)
    if hasattr(config, "hybrid_composition") and hasattr(
        nsos, "HybridComposition"
    ):
        config.hybrid_composition = nsos.HybridComposition.PARALLEL_GATED

    model = nsos.JambaModel(config, nsos.Device.GPU)
    model.to(nsos.Device.GPU)
    return model


def train_one(
    nsos: Any,
    dataset: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = build_mamba_only(
        nsos, dataset["vocab_size"], seed, args.deterministic
    )
    initial_manifest = product_benchmark.parameter_manifest(model)
    trainer = nsos.Trainer(model, args.learning_rate)
    schedule_steps = args.schedule_steps or args.steps
    trainer.warmup_steps = min(100, max(1, schedule_steps // 10))
    trainer.total_training_steps = schedule_steps
    trainer.weight_decay = 0.01
    trainer.max_grad_norm = 1.0

    schedule = nsos.TrainPhaseScheduler()
    schedule.progressive_qat_enabled = False
    schedule.semantic_warmup_steps = min(
        100, max(1, schedule_steps // 10)
    )
    schedule.qat_start_step = min(300, max(2, schedule_steps // 5))
    schedule.quantized_precision_bits = 2
    schedule.ternary_regularization = 0.0
    schedule.auxiliary_stack_enabled = False
    schedule.auxiliary_session_adapt_enabled = False
    schedule.auxiliary_reasoning_enabled = False
    schedule.auxiliary_memory_enabled = False
    trainer.configure_progressive_qat(schedule)

    rng = random.Random(200000 + seed)
    model.set_training_mode(True)
    started = time.perf_counter()
    interval = max(1, args.steps // 5)
    interval_loss = 0.0
    loss_curve: list[dict[str, float | int]] = []
    for step in range(args.steps):
        indices = [
            rng.randrange(len(dataset["train"])) for _ in range(args.batch_size)
        ]
        prompts = [dataset["train"][index][0] for index in indices]
        answers = [[dataset["train"][index][1]] for index in indices]
        loss = float(trainer.train_supervised_batch(prompts, answers))
        if not math.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss for seed {seed} at step {step + 1}"
            )
        interval_loss += loss
        if (step + 1) % interval == 0:
            elapsed = time.perf_counter() - started
            mean_loss = interval_loss / interval
            loss_curve.append(
                {
                    "step": step + 1,
                    "mean_loss": mean_loss,
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"[mamba seed={seed}] {step + 1}/{args.steps} "
                f"loss~{mean_loss:.6f}",
                flush=True,
            )
            interval_loss = 0.0

    train_seconds = time.perf_counter() - started
    model.set_training_mode(False)
    accuracy, eval_seconds = product_benchmark.evaluate(
        model,
        dataset["test"],
        dataset["vocab_size"],
        args.eval_batch_size,
    )
    final_manifest = product_benchmark.parameter_manifest(model)
    return {
        "seed": seed,
        "data_seed": 200000 + seed,
        "accuracy": accuracy,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_seconds": train_seconds,
        "train_examples_per_second": (
            args.steps * args.batch_size / train_seconds
        ),
        "eval_seconds": eval_seconds,
        "last_objective_total": float(trainer.last_objective_stats.total),
        "loss_curve": loss_curve,
        "initial_parameter_manifest": initial_manifest,
        "final_parameter_manifest": final_manifest,
    }


def main() -> int:
    args = parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.repeats < 1
        or (args.schedule_steps is not None and args.schedule_steps < args.steps)
    ):
        raise ValueError(
            "steps, batch-size, and repeats must be positive and "
            "schedule-steps must be >= steps"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    args.build_dir = args.build_dir.resolve()
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    nsos = product_benchmark.load_nsos(args.build_dir)
    nsos.set_strict_gpu_execution(True)
    nsos.set_matmul_precision("fp32")
    dataset = product_benchmark.load_babi(args.data_dir)
    binary = extension_path(nsos)

    print(
        f"extension={binary} sha256={sha256(binary)} "
        f"backend={nsos.gpu_backend_name()} devices={list(nsos.gpu_devices())}",
        flush=True,
    )
    results = []
    for seed in args.seeds:
        for repeat in range(1, args.repeats + 1):
            result = train_one(nsos, dataset, seed, args)
            result["repeat"] = repeat
            results.append(result)
    report = {
        "schema_version": 1,
        "purpose": "protocol-identical Mamba-only quality regression diagnosis",
        "extension": {
            "path": str(binary),
            "sha256": sha256(binary),
            "size": binary.stat().st_size,
            "modified_ns": binary.stat().st_mtime_ns,
        },
        "backend": nsos.gpu_backend_name(),
        "deterministic": args.deterministic,
        "devices": list(nsos.gpu_devices()),
        "environment_controls": {
            name: os.environ.get(name)
            for name in (
                "NSOS_FUSED_OPT",
                "NSOS_MAMBA_WARP_AGGREGATE",
                "NSOS_MAMBA_REDUCED_CONV",
            )
        },
        "protocol": {
            "dataset": "facebook/babi_qa, en-10k-qa1",
            "revision": product_benchmark.BABI_REVISION,
            "hashes": dataset["hashes"],
            "train_examples": len(dataset["train"]),
            "test_examples": len(dataset["test"]),
            "vocab_size": dataset["vocab_size"],
            "max_length": dataset["max_length"],
            "steps": args.steps,
            "schedule_steps": args.schedule_steps or args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "repeats": args.repeats,
        },
        "results": results,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report={args.report} sha256={sha256(args.report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
