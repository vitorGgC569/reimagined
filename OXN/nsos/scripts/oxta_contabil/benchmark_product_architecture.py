"""Benchmark the supported NSOS product architecture on AMD HIP.

This benchmark intentionally keeps every experimental branch disabled.  It
compares Mamba-only, Attention-only, Mamba+Attention, and Mamba+Attention with
progressive 2-bit ternary QAT on the pinned bAbI QA1 protocol used by the
2026-07-22 T4 campaign.  A separate OxtaMem system arm measures structured
retrieval without pretending that retrieval is a model-only result.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_policy import (
    MAMBA_EXPAND,
    MAMBA_STATE,
    MODEL_LAYERS,
    MODEL_WIDTH,
    configured_test_inventory,
    resolve_gradient_checkpointing,
    validate_training_runtime_contract,
)


BABI_REVISION = "1d86ad39d1c3ea2ff4b77eeb85f7c6ebd622a95f"
BABI_HASHES = {
    "train.parquet": "f1d67aa230d7aba0ed310df0d696a3ba9a07270e1670fe64c6901c24e5016f3f",
    "test.parquet": "9875dfad271cbfa5c49adb5809dd67be3826394a5d4d66dc74cab0c81483e2f8",
}
TOKEN_RE = re.compile(r"[a-z]+|[?.]", re.I)
DEFAULT_ARMS = ("mamba", "attention", "hybrid", "hybrid_qat")
HISTORICAL_BASELINES = {
    "campaign": "T4 expanded validation, 2026-07-22",
    "protocol_compatibility": "same pinned bAbI QA1 split, 1500 steps, batch 32, seed 11",
    "majority": 0.187,
    "jamba_dense_seed11": 0.493,
    "jamba_moe_seed11_experimental": 0.271,
    "nsos_hybrid_seed11": 0.767,
    "nsos_hybrid_3seed_mean": 0.685,
    "nsos_hybrid_3seed_std_sample": 0.071,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=root / "build-codex-hip")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "artifacts" / "oxta_contabil_amd" / "data",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root
        / "artifacts"
        / "oxta_contabil_amd"
        / "benchmark"
        / "rx7600_product_architecture_babi.json",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "SSD history policy: auto retains histories when their worst-case "
            "footprint is <=20%% of selected GPU VRAM; on/off force a mode"
        ),
    )
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--hybrid-seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--ablation-seed", type=int, default=11)
    parser.add_argument(
        "--hybrid-audit-samples",
        type=int,
        default=4,
        help=(
            "post-training gradient probes used to diagnose Mamba/Attention "
            "dominance, redundancy and cancellation; 0 disables"
        ),
    )
    parser.add_argument(
        "--oxtamem-arm",
        choices=DEFAULT_ARMS,
        default="hybrid_qat",
        help="trained arm whose first seed is evaluated with OxtaMem",
    )
    parser.add_argument("--skip-oxtamem", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="directory for resumable model/trainer checkpoints (default: beside report)",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="fail before training when tracked or untracked workspace changes exist",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_sha256_f32_le(values: np.ndarray, shape: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"NSOS-TENSOR-F32-LE-v1\x00")
    digest.update(struct.pack("<Q", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def parameter_manifest(model) -> dict[str, Any]:
    rows = []
    names: set[str] = set()
    for index, parameter in enumerate(model.parameters()):
        name = str(parameter.name)
        if not name or name in names:
            raise RuntimeError(f"missing/duplicate parameter name: {name!r}")
        names.add(name)
        shape = [int(value) for value in parameter.data.shape]
        values = np.asarray(
            parameter.data.numpy(), dtype="<f4"
        ).reshape(-1)
        rows.append(
            {
                "registry_index": index,
                "name": name,
                "base_name": str(parameter.base_name),
                "shape": shape,
                "elements": int(parameter.data.size),
                "trainable": bool(parameter.trainable),
                "sha256": tensor_sha256_f32_le(values, shape),
            }
        )
    aliases = [
        {"alias": str(alias), "canonical": str(canonical)}
        for alias, canonical in model.parameter_aliases()
    ]
    alias_names: set[str] = set()
    for alias in aliases:
        if (
            not alias["alias"]
            or alias["alias"] in names
            or alias["alias"] in alias_names
            or alias["canonical"] not in names
        ):
            raise RuntimeError(
                f"invalid parameter alias contract: {alias!r}"
            )
        alias_names.add(alias["alias"])
    topology_payload = [
        {
            "name": row["name"],
            "shape": row["shape"],
            "trainable": row["trainable"],
        }
        for row in rows
    ]
    topology_payload.extend(
        {
            "alias": alias["alias"],
            "canonical": alias["canonical"],
        }
        for alias in aliases
    )
    value_payload = [
        {"name": row["name"], "sha256": row["sha256"]} for row in rows
    ]
    return {
        "hash_format": "sha256/nsos-tensor-f32-le-v1",
        "registry_tensors": len(rows),
        "logical_parameter_paths": len(rows) + len(aliases),
        "registry_elements": sum(row["elements"] for row in rows),
        "trainable_tensors": sum(row["trainable"] for row in rows),
        "trainable_elements": sum(
            row["elements"] for row in rows if row["trainable"]
        ),
        "topology_sha256": hashlib.sha256(
            json.dumps(
                topology_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "values_sha256": hashlib.sha256(
            json.dumps(
                value_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "aliases": aliases,
        "parameters": rows,
    }


def git_provenance(root: Path, require_clean: bool) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError(
            "official benchmark requires a clean Git workspace"
        )
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def effective_topology(config, arm: str) -> list[dict[str, Any]]:
    layers = []
    for layer in range(int(config.num_layers)):
        scheduled = (
            layer % int(config.attention_period)
        ) == int(config.attention_slot)
        guarded = bool(
            scheduled
            and config.force_mamba_last_layer
            and layer + 1 == int(config.num_layers)
        )
        if not scheduled or guarded:
            block_type = "mamba"
        elif arm == "attention":
            block_type = "attention+ffn"
        else:
            block_type = "mamba+attention+ffn"
        layers.append(
            {
                "index": layer,
                "scheduled_attention": scheduled,
                "last_layer_guard_applied": guarded,
                "effective_type": block_type,
            }
        )
    return layers


def load_nsos(build_dir: Path):
    build_dir = build_dir.resolve()
    if os.name == "nt":
        runtime_dirs = (
            Path(r"C:\TheRock\build\bin"),
            Path(r"C:\TheRock\build\lib\llvm\bin"),
            build_dir,
        )
        # The handles must stay alive while nsos_ext is loaded.
        load_nsos.dll_handles = [
            os.add_dll_directory(str(path)) for path in runtime_dirs if path.is_dir()
        ]
    sys.path.insert(0, str(build_dir))
    import nsos_ext as nsos  # type: ignore

    return nsos


load_nsos.dll_handles = []


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def unpack_babi(frame: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for story in frame["story"]:
        context: list[str] = []
        for kind, text, answer, supporting in zip(
            story["type"], story["text"], story["answer"], story["supporting_ids"]
        ):
            if int(kind) == 0:
                context.append(str(text))
            else:
                examples.append(
                    {
                        "context": list(context),
                        "question": str(text),
                        "answer": str(answer),
                        "supporting": [str(item) for item in supporting],
                    }
                )
    return examples


def load_babi(data_dir: Path) -> dict[str, Any]:
    babi_dir = data_dir.resolve() / "babi_qa1"
    paths = {split: babi_dir / f"{split}.parquet" for split in ("train", "test")}
    for path in paths.values():
        actual = sha256(path)
        expected = BABI_HASHES[path.name]
        if actual != expected:
            raise RuntimeError(f"bAbI hash mismatch for {path}: {actual} != {expected}")

    train = unpack_babi(pd.read_parquet(paths["train"]))
    test = unpack_babi(pd.read_parquet(paths["test"]))
    special = ["<pad>", "<unk>", "<sep>", "<answer>"]
    train_tokens: set[str] = set()
    for example in train:
        for sentence in example["context"]:
            train_tokens.update(words(sentence))
        train_tokens.update(words(example["question"]))
        train_tokens.add(example["answer"].lower())
    vocab = special + sorted(train_tokens)
    stoi = {token: index for index, token in enumerate(vocab)}

    def encode(example: dict[str, Any]) -> tuple[list[int], int]:
        ids: list[int] = []
        for sentence in example["context"]:
            ids.extend(stoi.get(token, 1) for token in words(sentence))
            ids.append(stoi["<sep>"])
        ids.extend(stoi.get(token, 1) for token in words(example["question"]))
        ids.append(stoi["<answer>"])
        return ids, stoi.get(example["answer"].lower(), 1)

    train_encoded = [encode(example) for example in train]
    test_encoded = [encode(example) for example in test]
    max_length = max(len(ids) for ids, _ in train_encoded + test_encoded)

    def pad(item: tuple[list[int], int]) -> tuple[list[int], int]:
        ids, answer = item
        return [stoi["<pad>"]] * (max_length - len(ids)) + ids, answer

    train_padded = [pad(item) for item in train_encoded]
    test_padded = [pad(item) for item in test_encoded]
    counts = collections.Counter(answer for _, answer in test_padded)
    return {
        "train_examples": train,
        "test_examples": test,
        "train": train_padded,
        "test": test_padded,
        "vocab": vocab,
        "stoi": stoi,
        "vocab_size": len(vocab),
        "max_length": max_length,
        "majority_accuracy": max(counts.values()) / len(test_padded),
        "hashes": {path.name: sha256(path) for path in paths.values()},
    }


def build_model(
    nsos,
    arm: str,
    vocab_size: int,
    seed: int,
    use_gradient_checkpointing: bool,
):
    nsos.set_seed(seed)
    nsos.set_deterministic_reductions(True)
    if not nsos.deterministic_reductions_enabled():
        raise RuntimeError(
            "product benchmark requires deterministic GPU reductions"
        )
    config = nsos.ModelConfig()
    config.architecture_schema_version = 2
    config.num_layers = MODEL_LAYERS
    config.d_model = MODEL_WIDTH
    config.vocab_size = vocab_size
    config.n_heads = 4
    config.n_kv_heads = 2
    config.max_context_tokens = 128
    config.default_batch_size = 32
    config.use_cuda = True
    config.dropout = 0.0
    config.use_moe = False
    config.use_kan = False
    config.use_ttt = False
    config.use_chrass = False
    config.use_slender_embedding = False
    config.mamba2_faithful = True
    config.mamba_expand = MAMBA_EXPAND
    config.mamba_head_dim = 64
    config.mamba_state_expansion = True
    config.mamba_d_state = MAMBA_STATE
    config.mamba_n_groups = 1
    config.use_gradient_checkpointing = use_gradient_checkpointing
    config.use_flash_attn = False
    config.force_mamba_last_layer = False
    config.faithful_attention_linears = True
    config.hybrid_mamba_gate_init = 1.0
    config.hybrid_attention_gate_init = 0.01
    config.hybrid_ffn_gate_init = 0.01
    if arm == "mamba":
        config.hybrid_composition = nsos.HybridComposition.PARALLEL_GATED
        config.attention_period = 64
        config.attention_slot = 63
        config.use_exact_attention_training = False
    elif arm == "attention":
        # Attention-only is an explicit replacement ablation. Under the
        # production ParallelGated contract, period=1 would be hybrid at every
        # layer and would invalidate this control arm.
        config.hybrid_composition = nsos.HybridComposition.LEGACY_REPLACEMENT
        config.attention_period = 1
        config.attention_slot = 0
        config.use_exact_attention_training = True
    else:
        config.hybrid_composition = nsos.HybridComposition.PARALLEL_GATED
        config.attention_period = 2
        config.attention_slot = 1
        config.use_exact_attention_training = True
    model = nsos.JambaModel(config, nsos.Device.GPU)
    model.to(nsos.Device.GPU)
    return model, config


def evaluate(model, encoded: list[tuple[list[int], int]], vocab_size: int, batch: int):
    started = time.perf_counter()
    correct = 0
    for offset in range(0, len(encoded), batch):
        rows = encoded[offset : offset + batch]
        sequences = [sequence for sequence, _ in rows]
        logits = np.asarray(model.forward_ids_batch(sequences).numpy())
        logits = logits.reshape(len(rows), len(sequences[0]), vocab_size)
        predicted = logits[:, -1, :].argmax(axis=-1)
        correct += sum(int(int(pred) == target) for pred, (_, target) in zip(predicted, rows))
    elapsed = time.perf_counter() - started
    return correct / len(encoded), elapsed


def collect_hybrid_diagnostics(
    nsos,
    model,
    trainer,
    dataset,
    arm: str,
    seed: int,
    sample_count: int,
    output_path: Path,
) -> dict[str, Any] | None:
    if arm not in {"hybrid", "hybrid_qat"} or sample_count <= 0:
        return None
    count = min(sample_count, len(dataset["train"]))
    if count == 0:
        raise RuntimeError("hybrid audit requires at least one training row")

    collector = nsos.LayerAuditCollector()
    collector.begin_run(f"{arm}.seed-{seed}.post-train")
    collector.set_phase("post_train_gradient_probe")
    collector.set_storage_policy(False, 1, 10000, False)
    collector.set_parameter_audit_policy(False)
    model.set_audit_collector(collector)
    collector.set_enabled(True)
    probe_losses: list[float] = []
    try:
        # Use deterministic, evenly spread examples. Appending the answer token
        # creates the next-token objective accepted by accumulate_gradients,
        # while deliberately avoiding an optimizer step.
        for probe_index in range(count):
            row_index = probe_index * len(dataset["train"]) // count
            prompt, answer = dataset["train"][row_index]
            collector.set_step(probe_index + 1)
            probe_loss = float(
                trainer.accumulate_gradients([*prompt, int(answer)], [])
            )
            if not math.isfinite(probe_loss):
                raise RuntimeError(
                    f"non-finite hybrid diagnostic loss for {arm} seed {seed}"
                )
            probe_losses.append(probe_loss)
    finally:
        collector.set_enabled(False)
        for parameter in model.parameters():
            parameter.zero_grad()
        model.set_training_mode(False)

    records = list(collector.hybrid_interaction_records())
    if not records:
        raise RuntimeError(
            f"{arm} produced no hybrid interaction audit records"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collector.write_json(str(output_path))

    serialized_records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Any]] = {}
    for record in records:
        key = (str(record.pass_name), int(record.layer_index))
        grouped.setdefault(key, []).append(record)
        serialized_records.append(
            {
                "sequence": int(record.sequence),
                "pass": key[0],
                "step": int(record.step),
                "layer_index": key[1],
                "signal_cosine": float(record.signal_cosine),
                "contribution_cosine": float(
                    record.contribution_cosine
                ),
                "attention_to_mamba_signal_ratio": float(
                    record.attention_to_mamba_signal_ratio
                ),
                "attention_to_mamba_contribution_ratio": float(
                    record.attention_to_mamba_contribution_ratio
                ),
                "cancellation_fraction": float(
                    record.cancellation_fraction
                ),
            }
        )

    layer_pass_summary: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for (pass_name, layer_index), rows in sorted(grouped.items()):
        signal_cosine = statistics.fmean(
            float(row.signal_cosine) for row in rows
        )
        contribution_cosine = statistics.fmean(
            float(row.contribution_cosine) for row in rows
        )
        ratio = statistics.fmean(
            float(row.attention_to_mamba_contribution_ratio)
            for row in rows
        )
        cancellation = statistics.fmean(
            float(row.cancellation_fraction) for row in rows
        )
        layer_pass_summary.append(
            {
                "pass": pass_name,
                "layer_index": layer_index,
                "samples": len(rows),
                "signal_cosine_mean": signal_cosine,
                "contribution_cosine_mean": contribution_cosine,
                "attention_to_mamba_contribution_ratio_mean": ratio,
                "cancellation_fraction_mean": cancellation,
            }
        )
        if ratio >= 4.0:
            findings.append(
                {
                    "pass": pass_name,
                    "layer_index": layer_index,
                    "kind": "attention_dominates",
                    "value": ratio,
                }
            )
        elif ratio <= 0.25:
            findings.append(
                {
                    "pass": pass_name,
                    "layer_index": layer_index,
                    "kind": "attention_suppressed",
                    "value": ratio,
                }
            )
        if contribution_cosine >= 0.8:
            findings.append(
                {
                    "pass": pass_name,
                    "layer_index": layer_index,
                    "kind": "branch_redundancy",
                    "value": contribution_cosine,
                }
            )
        elif contribution_cosine <= -0.5:
            findings.append(
                {
                    "pass": pass_name,
                    "layer_index": layer_index,
                    "kind": "branch_antagonism",
                    "value": contribution_cosine,
                }
            )
        if cancellation >= 0.25:
            findings.append(
                {
                    "pass": pass_name,
                    "layer_index": layer_index,
                    "kind": "residual_cancellation",
                    "value": cancellation,
                }
            )

    return {
        "schema_version": 1,
        "audit_path": str(output_path.resolve()),
        "audit_sha256": sha256(output_path),
        "probe_samples": count,
        "probe_loss_mean": statistics.fmean(probe_losses),
        "interaction_records": serialized_records,
        "layer_pass_summary": layer_pass_summary,
        "findings": findings,
        "thresholds": {
            "dominance_ratio": 4.0,
            "suppression_ratio": 0.25,
            "redundancy_cosine": 0.8,
            "antagonism_cosine": -0.5,
            "cancellation_fraction": 0.25,
        },
    }


def train_arm(nsos, dataset, arm: str, seed: int, args: argparse.Namespace):
    model, config = build_model(
        nsos,
        arm,
        dataset["vocab_size"],
        seed,
        bool(args.checkpoint_policy["enabled"]),
    )
    initial_manifest = parameter_manifest(model)
    trainer = nsos.Trainer(model, args.learning_rate)
    trainer.warmup_steps = min(100, max(1, args.steps // 10))
    trainer.total_training_steps = args.steps
    trainer.weight_decay = 0.01
    trainer.max_grad_norm = 1.0
    schedule = nsos.TrainPhaseScheduler()
    schedule.progressive_qat_enabled = arm == "hybrid_qat"
    schedule.semantic_warmup_steps = min(100, max(1, args.steps // 10))
    schedule.qat_start_step = min(300, max(2, args.steps // 5))
    schedule.quantized_precision_bits = 2
    schedule.ternary_regularization = 1e-3 if arm == "hybrid_qat" else 0.0
    schedule.auxiliary_stack_enabled = False
    schedule.auxiliary_session_adapt_enabled = False
    schedule.auxiliary_reasoning_enabled = False
    schedule.auxiliary_memory_enabled = False
    trainer.configure_progressive_qat(schedule)

    rng = random.Random(200000 + seed)
    nsos.reset_gpu_transfer_stats()
    model.set_training_mode(True)
    started = time.perf_counter()
    interval = max(1, args.steps // 5)
    recent_loss = 0.0
    loss_curve: list[dict[str, Any]] = []
    for step in range(args.steps):
        indices = [rng.randrange(len(dataset["train"])) for _ in range(args.batch_size)]
        prompts = [dataset["train"][index][0] for index in indices]
        answers = [[dataset["train"][index][1]] for index in indices]
        loss = float(trainer.train_supervised_batch(prompts, answers))
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {arm} seed {seed} at step {step + 1}")
        recent_loss += loss
        if (step + 1) % interval == 0:
            elapsed = time.perf_counter() - started
            mean_loss = recent_loss / interval
            loss_curve.append(
                {
                    "step": step + 1,
                    "mean_loss": mean_loss,
                    "elapsed_seconds": elapsed,
                    "examples_per_second": (
                        (step + 1) * args.batch_size / elapsed
                    ),
                }
            )
            print(
                f"[{arm} seed={seed}] {step + 1}/{args.steps} "
                f"loss~{mean_loss:.4f}",
                flush=True,
            )
            recent_loss = 0.0
    train_seconds = time.perf_counter() - started
    training_transfer_stats = dict(nsos.gpu_transfer_stats())
    training_runtime_telemetry = dict(model.runtime_telemetry())
    training_pool_stats = dict(nsos.pool_stats())
    training_runtime_contract = validate_training_runtime_contract(
        training_runtime_telemetry,
        training_transfer_stats,
        training_pool_stats,
        arm,
        args.steps,
        bool(args.checkpoint_policy["enabled"]),
        deterministic_reductions=bool(
            nsos.deterministic_reductions_enabled()
        ),
    )
    nsos.reset_gpu_transfer_stats()
    model.set_training_mode(False)
    accuracy, eval_seconds = evaluate(
        model, dataset["test"], dataset["vocab_size"], args.eval_batch_size
    )
    evaluation_transfer_stats = dict(nsos.gpu_transfer_stats())
    # Snapshot the objective before the post-training gradient probe changes
    # last_objective_stats. Runtime/pool/transfer telemetry was captured at the
    # training boundary above and therefore excludes evaluation as promised.
    training_last_objective_total = float(
        trainer.last_objective_stats.total
    )
    final_manifest = parameter_manifest(model)
    checkpoint_stem = f"{arm}.seed-{seed}.step-{args.steps}"
    model_checkpoint = args.checkpoint_dir / f"{checkpoint_stem}.model.bin"
    trainer_checkpoint = args.checkpoint_dir / f"{checkpoint_stem}.trainer.bin"
    # JambaModel::save publishes through a same-directory unique temporary
    # file and an atomic replace. Avoid an outer fixed ".tmp" name that could
    # collide with another benchmark process.
    model.save(str(model_checkpoint))
    trainer.save_training_state(
        str(trainer_checkpoint), str(model_checkpoint)
    )
    checkpoint_hashes = {
        "model_path": str(model_checkpoint.resolve()),
        "trainer_path": str(trainer_checkpoint.resolve()),
        "model_sha256": sha256(model_checkpoint),
        "trainer_sha256": sha256(trainer_checkpoint),
    }
    # The diagnostic performs backward without an optimizer update. Run it
    # only after publishing both checkpoints so neither gradients nor trainer
    # bookkeeping from the probe can contaminate the resumable training state.
    hybrid_diagnostics = collect_hybrid_diagnostics(
        nsos,
        model,
        trainer,
        dataset,
        arm,
        seed,
        args.hybrid_audit_samples,
        args.checkpoint_dir
        / f"{arm}.seed-{seed}.step-{args.steps}.hybrid-audit.json",
    )
    result = {
        "arm": arm,
        "seed": seed,
        "data_seed": 200000 + seed,
        "accuracy": accuracy,
        "registry_parameters": final_manifest["registry_elements"],
        "trainable_parameters": final_manifest["trainable_elements"],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_seconds": train_seconds,
        "train_steps_per_second": args.steps / train_seconds,
        "train_examples_per_second": args.steps * args.batch_size / train_seconds,
        "eval_seconds": eval_seconds,
        "eval_examples_per_second": len(dataset["test"]) / eval_seconds,
        "qat_active_at_end": bool(trainer.progressive_qat_active()),
        "last_objective_total": training_last_objective_total,
        "loss_curve": loss_curve,
        "runtime_telemetry": training_runtime_telemetry,
        "gpu_transfer_stats": training_transfer_stats,
        "evaluation_gpu_transfer_stats": evaluation_transfer_stats,
        "gpu_pool_stats": training_pool_stats,
        "training_runtime_contract": training_runtime_contract,
        "hybrid_diagnostics": hybrid_diagnostics,
        "initial_parameter_manifest": initial_manifest,
        "final_parameter_manifest": final_manifest,
        "checkpoint": checkpoint_hashes,
        "effective_topology": effective_topology(config, arm),
        "config": {
            "layers": config.num_layers,
            "d_model": config.d_model,
            "architecture_schema_version": config.architecture_schema_version,
            "attention_period": config.attention_period,
            "attention_slot": config.attention_slot,
            "hybrid_composition": (
                "parallel_gated"
                if arm in {"mamba", "hybrid", "hybrid_qat"}
                else "legacy_replacement_attention_ablation"
            ),
            "force_mamba_last_layer": config.force_mamba_last_layer,
            "faithful_attention_linears": config.faithful_attention_linears,
            "mamba_d_state": config.mamba_d_state,
            "deterministic_reductions": (
                nsos.deterministic_reductions_enabled()
            ),
            "use_gradient_checkpointing": config.use_gradient_checkpointing,
            "use_moe": config.use_moe,
            "use_kan": config.use_kan,
            "use_ttt": config.use_ttt,
            "use_chrass": config.use_chrass,
            "use_slender_embedding": config.use_slender_embedding,
            "quantized_precision_bits": schedule.quantized_precision_bits,
        },
    }
    print(f"RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    return model, trainer, result


class OxtaMem:
    def __init__(self, library: Path, store: Path, size_mb: int = 16):
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.lib.oxtamem_abi_version.restype = ctypes.c_uint32
        if self.lib.oxtamem_abi_version() != 2:
            raise RuntimeError("OxtaMem ABI v2 is required")
        self.lib.oxtamem_create.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        self.lib.oxtamem_create.restype = ctypes.c_void_p
        self.lib.oxtamem_destroy.argtypes = [ctypes.c_void_p]
        self.lib.oxtamem_write_with_vector.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.oxtamem_write_with_vector.restype = ctypes.c_bool
        self.lib.oxtamem_search_similar.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.oxtamem_search_similar.restype = ctypes.c_bool
        self.lib.oxtamem_free_buffer.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.handle = self.lib.oxtamem_create(str(store).encode("utf-8"), size_mb)
        if not self.handle:
            raise RuntimeError("could not create OxtaMem store")

    def close(self):
        if self.handle:
            self.lib.oxtamem_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def write(self, key: str, payload: bytes, vector: np.ndarray):
        data = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        values = np.ascontiguousarray(vector, dtype=np.float32)
        ok = self.lib.oxtamem_write_with_vector(
            self.handle,
            key.encode(),
            data,
            len(payload),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.size,
        )
        if not ok:
            raise RuntimeError("OxtaMem vector write failed")

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[bytes]:
        values = np.ascontiguousarray(vector, dtype=np.float32)
        output = ctypes.POINTER(ctypes.c_uint8)()
        length = ctypes.c_size_t()
        ok = self.lib.oxtamem_search_similar(
            self.handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.size,
            top_k,
            ctypes.byref(output),
            ctypes.byref(length),
        )
        if not ok:
            raise RuntimeError("OxtaMem similarity search failed")
        blob = ctypes.string_at(output, length.value)
        self.lib.oxtamem_free_buffer(output, length.value)
        if len(blob) < 8:
            raise RuntimeError("truncated OxtaMem search payload")
        count = int.from_bytes(blob[:8], "little")
        if count > top_k:
            raise RuntimeError("OxtaMem search returned more rows than requested")
        cursor = 8
        result: list[bytes] = []
        for _ in range(count):
            if cursor + 8 > len(blob):
                raise RuntimeError("truncated OxtaMem result length")
            size = int.from_bytes(blob[cursor : cursor + 8], "little")
            cursor += 8
            if size > len(blob) - cursor:
                raise RuntimeError("truncated OxtaMem result payload")
            result.append(blob[cursor : cursor + size])
            cursor += size
        if cursor != len(blob):
            raise RuntimeError("malformed OxtaMem search payload")
        return result


def key_vector(example_id: int, entity: str) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(f"babi-qa1:{example_id}:{entity}".encode()).digest()[:8],
        "little",
    )
    vector = np.random.default_rng(seed).standard_normal(128).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def run_oxtamem_arm(model, dataset, library: Path, report_dir: Path, eval_batch: int):
    stoi = dataset["stoi"]

    def entity(text: str, question: bool) -> str:
        tokens = words(text)
        required = 3 if question else 1
        if len(tokens) < required:
            raise RuntimeError(
                f"unexpected bAbI entity sentence: {text!r}"
            )
        return tokens[2] if question else tokens[0]

    with tempfile.TemporaryDirectory(prefix="oxtamem_babi_", dir=report_dir) as temporary:
        with OxtaMem(library, Path(temporary) / "babi.db") as store:
            expected: dict[tuple[int, str], bytes] = {}
            for example_id, example in enumerate(dataset["test_examples"]):
                latest = {
                    entity(sentence, False): sentence
                    for sentence in example["context"]
                }
                for name, sentence in latest.items():
                    token_ids = [
                        stoi.get(token, stoi["<unk>"])
                        for token in words(sentence)
                    ]
                    payload = np.asarray(
                        token_ids, dtype="<u4"
                    ).tobytes(order="C")
                    expected[(example_id, name)] = payload
                    store.write(
                        f"babi-{example_id}-{name}",
                        payload,
                        key_vector(example_id, name),
                    )

            exact_hits = 0
            noisy_hits = 0
            prompts: list[list[int]] = []
            targets: list[int] = []
            noisy_rng = np.random.default_rng(20260728)
            for example_id, example in enumerate(dataset["test_examples"]):
                name = entity(example["question"], True)
                vector = key_vector(example_id, name)
                exact = store.search(vector, 1)
                exact_payload = exact[0] if exact else b""
                exact_hits += int(
                    exact_payload == expected[(example_id, name)]
                )
                noisy = vector + noisy_rng.normal(
                    0.0, 0.05, vector.shape
                ).astype(np.float32)
                noisy /= np.linalg.norm(noisy) + 1e-8
                noisy_result = store.search(noisy, 1)
                noisy_hits += int(
                    bool(noisy_result)
                    and noisy_result[0] == expected[(example_id, name)]
                )
                if len(exact_payload) % 4 != 0:
                    raise RuntimeError(
                        "OxtaMem token payload is not aligned uint32"
                    )
                retrieved_ids = np.frombuffer(
                    exact_payload, dtype="<u4"
                ).astype(np.int64).tolist()
                question = [
                    stoi.get(token, stoi["<unk>"])
                    for token in words(example["question"])
                ]
                prompt = (
                    retrieved_ids
                    + [stoi["<sep>"]]
                    + question
                    + [stoi["<answer>"]]
                )
                if len(prompt) > dataset["max_length"]:
                    prompt = prompt[-dataset["max_length"] :]
                prompt = [stoi["<pad>"]] * (
                    dataset["max_length"] - len(prompt)
                ) + prompt
                prompts.append(prompt)
                targets.append(stoi[example["answer"].lower()])

        encoded = list(zip(prompts, targets))
        system_accuracy, eval_seconds = evaluate(
            model, encoded, dataset["vocab_size"], eval_batch
        )
    total = len(dataset["test_examples"])
    return {
        "retrieval_at_1_exact": exact_hits / total,
        "retrieval_at_1_noisy_sigma_0_05": noisy_hits / total,
        "system_accuracy_with_exact_retrieval": system_accuracy,
        "eval_seconds": eval_seconds,
        "test_examples": total,
        "retrieval_protocol": "entity-keyed deterministic vectors; no answer/supporting-id leakage",
        "interpretation": "system integration result, not model-only architecture evidence",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in DEFAULT_ARMS:
        rows = [row for row in results if row["arm"] == arm]
        if not rows:
            continue
        accuracies = [row["accuracy"] for row in rows]
        summary[arm] = {
            "runs": len(rows),
            "seeds": [row["seed"] for row in rows],
            "accuracy_mean": statistics.fmean(accuracies),
            "accuracy_std_sample": (
                statistics.stdev(accuracies) if len(accuracies) > 1 else None
            ),
            "accuracy_values": accuracies,
            "train_seconds_mean": statistics.fmean(row["train_seconds"] for row in rows),
            "train_examples_per_second_mean": statistics.fmean(
                row["train_examples_per_second"] for row in rows
            ),
        }
    return summary


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("benchmark arms must be unique")
    if len(set(args.hybrid_seeds)) != len(args.hybrid_seeds):
        raise ValueError("hybrid seeds must be unique")
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else args.report.parent / "checkpoints"
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[4]
    provenance = git_provenance(
        repository_root, args.require_clean_git
    )
    test_inventory = configured_test_inventory(args.build_dir)
    nsos = load_nsos(args.build_dir)
    nsos.set_strict_gpu_execution(True)
    nsos.set_matmul_precision("fp32")
    dataset = load_babi(args.data_dir)
    devices = list(nsos.gpu_devices())
    if test_inventory["gpu_backend"].lower() != nsos.gpu_backend_name():
        raise RuntimeError(
            "configured CTest inventory backend does not match the loaded "
            "NSOS extension"
        )
    eligible_devices = [
        device
        for device in devices
        if bool(device.get("compiled", False))
        and not bool(device.get("integrated", False))
    ]
    if not eligible_devices:
        raise RuntimeError(
            "product benchmark requires a visible discrete GPU whose "
            "architecture is compiled into the active NSOS binary"
        )
    args.checkpoint_policy = resolve_gradient_checkpointing(
        args.gradient_checkpointing,
        args.batch_size,
        dataset["max_length"],
        devices,
    )
    print(
        f"GPU backend={nsos.gpu_backend_name()} vendor={nsos.gpu_vendor_name()} "
        f"devices={devices}",
        flush=True,
    )
    print(
        f"bAbI train={len(dataset['train'])} test={len(dataset['test'])} "
        f"vocab={dataset['vocab_size']} maxlen={dataset['max_length']}",
        flush=True,
    )
    print(
        "gradient_checkpointing="
        f"{args.checkpoint_policy['enabled']} "
        f"reason={args.checkpoint_policy['reason']} "
        "estimated_history_bytes="
        f"{args.checkpoint_policy['estimated_history_bytes']}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    oxtamem_result = None
    partial_report = args.report.with_suffix(".partial.json")
    for arm in args.arms:
        seeds = args.hybrid_seeds if arm in {"hybrid", "hybrid_qat"} else [args.ablation_seed]
        for seed in seeds:
            model, trainer, result = train_arm(nsos, dataset, arm, seed, args)
            results.append(result)
            oxtamem_seed = (
                args.hybrid_seeds[0]
                if arm in {"hybrid", "hybrid_qat"}
                else args.ablation_seed
            )
            if (
                arm == args.oxtamem_arm
                and seed == oxtamem_seed
                and not args.skip_oxtamem
            ):
                library = (
                    Path(__file__).resolve().parents[4]
                    / "modules"
                    / "oxtamem"
                    / "oxta_engine"
                    / "target"
                    / "release"
                    / ("oxta_mem.dll" if os.name == "nt" else "liboxta_mem.so")
                )
                oxtamem_result = run_oxtamem_arm(
                    model, dataset, library, args.report.parent, args.eval_batch_size
                )
                oxtamem_result["model_arm"] = arm
                oxtamem_result["model_seed"] = seed
                print(f"OXTAMEM {json.dumps(oxtamem_result, sort_keys=True)}", flush=True)
            atomic_write_json(
                partial_report,
                {
                    "schema_version": 2,
                    "complete": False,
                    "provenance": provenance,
                    "configured_test_inventory": test_inventory,
                    "results": results,
                    "summary": summarize(results),
                    "oxtamem": oxtamem_result,
                },
            )
            del trainer, model
            gc.collect()
            nsos.release_cached_memory()

    report = {
        "schema_version": 2,
        "complete": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "AMD product architecture ablation; experimental branches disabled",
        "provenance": provenance,
        "configured_test_inventory": test_inventory,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "nsos_environment": dict(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if key.startswith("NSOS_")
                )
            ),
        },
        "hardware": {
            "backend": nsos.gpu_backend_name(),
            "vendor": nsos.gpu_vendor_name(),
            "devices": devices,
            "strict_gpu_execution": nsos.strict_gpu_execution(),
            "matmul_precision": "fp32",
        },
        "protocol": {
            "dataset": "facebook/babi_qa, en-10k-qa1",
            "revision": BABI_REVISION,
            "hashes": dataset["hashes"],
            "train_examples": len(dataset["train"]),
            "test_examples": len(dataset["test"]),
            "vocab_size_train_only": dataset["vocab_size"],
            "max_length": dataset["max_length"],
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seeds": {
                "ablation": args.ablation_seed,
                "hybrid": args.hybrid_seeds,
            },
            "oxtamem_arm": args.oxtamem_arm,
            "gradient_checkpointing": args.checkpoint_policy,
            "deterministic_reductions": True,
        },
        "architecture_policy": {
            "model": ["Mamba", "Attention", "2-bit ternary QAT"],
            "system_integration": ["OxtaMem"],
            "determinism": "required; fail closed",
            "explicitly_disabled": ["MoE", "KAN", "TTT", "CHRASS", "Slender embedding"],
        },
        "dataset_majority_accuracy": dataset["majority_accuracy"],
        "results": results,
        "summary": summarize(results),
        "oxtamem": oxtamem_result,
        "historical_baselines": {
            **HISTORICAL_BASELINES,
            "direct_architecture_comparison_valid": False,
            "reason": (
                "historical hybrid rows used attention replacement; the "
                "current hybrid is parallel-gated and preserves Mamba"
            ),
        },
    }
    atomic_write_json(args.report, report)
    partial_report.unlink(missing_ok=True)
    print(f"REPORT {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
