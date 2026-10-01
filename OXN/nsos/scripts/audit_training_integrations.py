#!/usr/bin/env python3
"""Small native GPU integration probe; one topology per isolated process.

An executable forward/backward is not proof of an exact gradient or useful
language learning. D2H gradient inspection happens AFTER performance counters.
"""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import train_ptbr_conversational as core
from bench_training_engineering import profile_environment, model_weight_fingerprint


def validate_probe_geometry(seq_len, steps, batch_size, d_model):
    if not 2 <= seq_len <= 8192 or steps < 2:
        raise ValueError("Require 2 <= seq-len <= 8192 and steps >= 2")
    if not 1 <= batch_size <= 8:
        raise ValueError("Require 1 <= batch-size <= 8")
    if not 64 <= d_model <= 768 or d_model % 64:
        raise ValueError("Require 64 <= d-model <= 768, divisible by 64")


def missing_gradient_paths(variant, parameters, layers=2):
    """Require actual task gradients, not just finite weights or weight decay."""
    required = ["embedding.weight", "norm_f.weight", ".mamba."]
    required.extend(f"layers.{i}." for i in range(layers))
    required.extend({
        "mamba": [], "attention": [".attn."], "parallel": [".attn.", ".ffn."],
        "moe": [".experts.", ".router."], "kan": [".kan."],
        "ttt": [".ttt.w_k.", ".ttt.w_v.", ".ttt.w_out."],
    }[variant])
    active = [p["name"] for p in parameters if p["gradient_present"] and p["gradient_nonzero"]]
    return [path for path in required if not any(path in name for name in active)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("mamba", "attention", "parallel", "moe", "kan", "ttt"), required=True)
    parser.add_argument("--redesign", action="store_true", help="Enable exact GPU training redesigns")
    parser.add_argument("--full-ttt", action="store_true", help="Versioned full-sequence TTT BPTT; only the TTT topology")
    parser.add_argument("--grouped-moe", action="store_true", help="Device-segmented sparse MoE training; only the MoE topology")
    parser.add_argument("--wmma-moe", action="store_true", help="Opt-in RDNA3 WMMA grouped MoE; implies --grouped-moe")
    parser.add_argument("--recompute-kan", action="store_true", help="Opt-in device QAT/implicit tiled RBF; KAN topology only")
    parser.add_argument("--wmma-kan", action="store_true", help="Opt-in RDNA3 implicit RBF WMMA; implies --recompute-kan")
    parser.add_argument("--batch-size", type=int, default=1, help="B>1 exercises heterogeneous masked SFT batches")
    parser.add_argument("--d-model", type=int, default=128, help="Model width; independent of the number of layers")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    args.grouped_moe = args.grouped_moe or args.wmma_moe
    args.recompute_kan = args.recompute_kan or args.wmma_kan
    try:
        validate_probe_geometry(args.seq_len, args.steps, args.batch_size, args.d_model)
    except ValueError as error:
        parser.error(str(error))
    if args.full_ttt and args.variant != "ttt":
        parser.error("--full-ttt requires --variant ttt")
    if args.grouped_moe and args.variant != "moe":
        parser.error("--grouped-moe requires --variant moe")
    if args.recompute_kan and args.variant != "kan":
        parser.error("--recompute-kan requires --variant kan")
    if args.output.exists():
        parser.error("Refusing to overwrite an integration result")
    for key in list(os.environ):
        if key.startswith("NSOS_") and key not in ("NSOS_HIP_ROOT", "NSOS_CUDA_ROOT", "NSOS_GPU_DEVICE"):
            del os.environ[key]
    os.environ.update(profile_environment("redesign-fp32" if args.redesign else "chunked-fp32"))
    nsos = core.load_nsos(args.build_dir)
    if not nsos.fast_gpu_supported():
        raise RuntimeError("GPU unavailable")
    nsos.set_seed(7301)
    nsos.set_deterministic_reductions(True)
    nsos.set_strict_gpu_execution(True)
    nsos.set_matmul_precision(args.precision)
    preset = dict(core.PRESETS["dryrun"], layers=2, d_model=args.d_model, seq_len=args.seq_len)
    config = core.build_model_config(nsos, preset, 257, nsos.Device.GPU)
    config.max_context_tokens = max(config.max_context_tokens, args.seq_len)
    config.sliding_window = config.max_context_tokens
    if args.variant in ("attention", "parallel"):
        config.attention_period = 2
        config.attention_slot = 0
    if args.variant == "attention":
        config.hybrid_composition = nsos.HybridComposition.LEGACY_REPLACEMENT
    if args.variant == "parallel":
        config.hybrid_composition = nsos.HybridComposition.PARALLEL_GATED
    if args.variant == "moe":
        config.use_moe = True
        config.moe_period = 2
        config.moe_slot = 0
        config.num_experts = 4
        config.num_experts_per_token = 2
        config.moe_expert_hidden_dim = args.d_model
    config.use_kan = args.variant == "kan"
    if args.variant == "ttt":
        config.use_ttt = True
        config.ttt_period = 2
        config.ttt_slot = 0
    args.gpu_training_profile = "redesign-v1" if args.redesign else "inherit"
    args.ttt_gradient_policy = "full-sequence-v1" if args.full_ttt else "truncated"
    args.moe_compute_policy = "wmma-v1" if args.wmma_moe else "grouped-v1" if args.grouped_moe else "legacy"
    args.kan_compute_policy = "wmma-v1" if args.wmma_kan else "tiled-v1" if args.recompute_kan else "legacy"
    selected = int(nsos.selected_gpu_device())
    wave = int(next(item["warp_size"] for item in nsos.gpu_devices() if int(item["index"]) == selected))
    training_policy = core.configure_gpu_training_profile(args, config, True, wave)
    config_snapshot = core.model_config_dict(config)
    # The production Python summary does not yet include all native hybrid
    # fields. Record them explicitly here so two topologies cannot look equal.
    config_snapshot.update({name: getattr(config, name) for name in (
        "architecture_schema_version", "faithful_attention_linears", "hybrid_mamba_gate_init",
        "hybrid_attention_gate_init", "hybrid_ffn_gate_init", "num_experts",
        "num_experts_per_token", "moe_expert_hidden_dim", "rope_theta")})
    config_snapshot["hybrid_composition"] = str(config.hybrid_composition)
    result = {"created_at": core.utc_now(), "variant": args.variant, "redesign": args.redesign,
              "batch_size": args.batch_size, "task": "causal" if args.batch_size == 1 else "masked_sft_batch",
              "seq_len": args.seq_len, "steps": args.steps, "precision": args.precision,
              "gpu_training_policy": training_policy,
              "config": config_snapshot, "strict_gpu": True,
              "deterministic": True, "binary_sha256": core.sha256_file(Path(nsos.__file__)),
              "probe_sha256": core.sha256_file(Path(__file__)),
              "core_script_sha256": core.sha256_file(Path(core.__file__)),
              "input_policy": "synthetic_ids_7301_causal_or_heterogeneous_sft_v1",
              "tokenizer": "not_used_synthetic_token_ids",
              "scope": "bounded synthetic-token integration and gradient coverage; not language-quality certification"}
    model = None
    try:
        model = nsos.JambaModel(config, nsos.Device.GPU)
        model.to(nsos.Device.GPU)
        trainer = nsos.Trainer(model, 0.0003)
        scheduler = trainer.phase_scheduler
        scheduler.progressive_qat_enabled = False
        scheduler.ternary_regularization = 0.0
        trainer.phase_scheduler = scheduler
        initial = {p.name: np.array(p.data.cpu().numpy(), copy=True) for p in model.parameters() if p.trainable}
        result["initial_weights"] = model_weight_fingerprint(model)
        result["execution_identity"] = dict(trainer.execution_identity())
        core.validate_native_training_policy(training_policy, result["execution_identity"])
        samples = []
        for step in range(args.steps):
            if step == 1:
                nsos.reset_gpu_transfer_stats()
            ids = [1+(step*11+i*7)%255 for i in range(args.seq_len + 1)]
            start = time.perf_counter()
            if args.batch_size == 1:
                loss = float(trainer.train_step(ids[:-1], ids[1:]))
            else:
                lengths = [max(2, args.seq_len - b * 7) for b in range(args.batch_size)]
                sequences = [[1 + (step * 11 + b * 13 + i * 7) % 255 for i in range(n)]
                             for b, n in enumerate(lengths)]
                prompts = [row[:len(row)//2] for row in sequences]
                answers = [row[len(row)//2:] for row in sequences]
                loss = float(trainer.train_supervised_batch(prompts, answers))
            samples.append({"loss": loss, "seconds": time.perf_counter()-start})
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite loss")
        result["samples"] = samples
        result["transfers_per_measured_step"] = {k:v/(args.steps-1) for k,v in nsos.gpu_transfer_stats().items()}
        result["runtime"] = dict(model.runtime_telemetry())
        result["dispatch_counters"] = dict(nsos.gpu_dispatch_counters())
        if args.grouped_moe and not result["dispatch_counters"].get("grouped_moe_training", 0):
            raise RuntimeError("Requested grouped MoE never executed")
        if args.recompute_kan and not result["dispatch_counters"].get("kan_recompute", 0):
            raise RuntimeError("Requested KAN recomputation never executed")
        if args.wmma_kan and args.precision != "fp32" and args.seq_len >= 16 and not result["dispatch_counters"].get("kan_wmma_gemm", 0):
            raise RuntimeError("Requested eligible KAN WMMA never dispatched")
        if args.wmma_moe and args.precision != "fp32" and args.seq_len >= 16 and not result["dispatch_counters"].get("grouped_moe_wmma_gemm", 0):
            raise RuntimeError("Requested eligible WMMA MoE never dispatched")
        result["resources"] = core.runtime_resource_snapshot(nsos, model, trainer)
        parameters = []
        for p in model.parameters():
            if not p.trainable:
                continue
            values = np.asarray(p.data.cpu().numpy())
            grad = np.asarray(p.grad.cpu().numpy()) if p.gradient_present else np.array([])
            parameters.append({"name": p.name, "elements": int(p.data.size),
                               "gradient_present": bool(p.gradient_present),
                               "gradient_storage_elements": int(p.grad.size),
                               "gradient_contributions_tracked": bool(p.gradient_contributions_tracked),
                               "gradient_nonzero": bool(np.any(grad != 0)),
                               "finite": bool(np.isfinite(values).all() and np.isfinite(grad).all()),
                               "max_abs_update": float(np.max(np.abs(values-initial[p.name])))})
        result["parameters"] = parameters
        result["final_weights"] = model_weight_fingerprint(model)
        result["missing_gradient_paths"] = missing_gradient_paths(args.variant, parameters)
        if result["missing_gradient_paths"]:
            raise RuntimeError("Missing active gradient paths: " + ", ".join(result["missing_gradient_paths"]))
        result["status"] = "executed_finite" if all(p["finite"] for p in parameters) else "nonfinite_parameters"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["transfers_at_failure"] = dict(nsos.gpu_transfer_stats())
    core.atomic_write_json(args.output.resolve(), result)
    print(f"[{args.variant}] {result['status']} {result.get('error', '')}", flush=True)
    return 0 if result["status"] == "executed_finite" else 1


if __name__ == "__main__":
    raise SystemExit(main())
