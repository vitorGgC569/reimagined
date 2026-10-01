#!/usr/bin/env python3
"""Bounded native HIP training probe; never writes production weights or packs.

Fresh process per profile is mandatory: native environment switches are cached.
This measures engine throughput/integration, not language quality or SFT readiness.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

import train_ptbr_conversational as core

PROFILES = ("baseline", "chunked-fp32", "chunked-bf16", "bf16-checkpoint",
            "boundary-fp32", "boundary-bf16", "clip-bf16", "redesign-fp32", "redesign-bf16")
TIMING_FIELDS = ("wall_ms", "preparation_ms", "inter_bucket_ms", "forward_ms",
                 "loss_ms", "backward_ms", "optimizer_ms", "unaccounted_ms")


def profile_environment(profile, timing=False, layer_timing=False):
    if profile not in PROFILES:
        raise ValueError("Unknown training profile")
    return {
        "NSOS_DETERMINISTIC": "1", "NSOS_TRAIN_CHUNK_SIZE": "1",
        "NSOS_TRAIN_TIMING": str(int(timing)),
        "NSOS_MAMBA_STAGE_TIMING": str(int(layer_timing)),
        "NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD": "0" if profile == "baseline" else "1",
        "NSOS_MAMBA_FORWARD_CHUNK_SIZE": "128",
        "NSOS_MAMBA_BACKWARD_CHUNK_SIZE": "128" if profile == "baseline" else "32",
        "NSOS_MAMBA_CHUNKED_BACKWARD": "1",
        "NSOS_MAMBA_STATE_MAJOR_HISTORY": "1",
        "NSOS_MAMBA_CHUNK_LDS_STATE_MAJOR": "1",
        "NSOS_MAMBA_PRECOMPUTE_DECAY": "1",
        "NSOS_MAMBA_BOUNDARY_HISTORY": str(int(profile.startswith(("boundary-", "redesign-")))),
        "NSOS_DEVICE_GRAD_CLIP": str(int(profile.startswith(("clip-", "redesign-")))),
        "NSOS_ATTN_TILED_TRAINING": str(int(profile.startswith("redesign-"))),
        "NSOS_MOE_ORDERED_DEVICE": str(int(profile.startswith("redesign-"))),
        "NSOS_TTT_DEVICE_RECURRENCE": str(int(profile.startswith("redesign-"))),
        "NSOS_TTT_FULL_BPTT": "0",
        "NSOS_MOE_GROUPED_TRAINING": "0",
        "NSOS_MOE_WMMA_TRAINING": "0",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
    }


def numeric_delta(after, before):
    return {k: v - before.get(k, 0) for k, v in after.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def model_weight_fingerprint(model):
    """Read-only download boundary, always outside the measured step interval."""
    digest = hashlib.sha256()
    elements = 0
    parameters = model.parameters()
    for parameter in parameters:
        values = np.asarray(parameter.data.cpu().numpy(), dtype="<f4")
        descriptor = json.dumps({"name": parameter.name, "shape": list(values.shape),
                                 "dtype": "little_endian_fp32"}, sort_keys=True).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(values.tobytes(order="C"))
        elements += values.size
    return {"sha256": digest.hexdigest(), "parameters": len(parameters), "elements": int(elements)}


def windows_adapter_residency():
    """WDDM aggregate gauges; not per-process allocation or hardware profiling."""
    if os.name != "nt":
        return {"status": "unavailable", "reason": "not Windows"}
    command = ("Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
               "-ErrorAction Stop | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted "
               "| ConvertTo-Json -Compress")
    try:
        process = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            return {"status": "unavailable", "reason": process.stderr[:256]}
        records = json.loads(process.stdout)
        if not records:
            return {"status": "unavailable", "reason": "no adapter gauges"}
        return {"status": "observed", "source": "WDDM aggregate; LUID mapping not certified",
                "adapters": records if isinstance(records, list) else [records]}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)[:256]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--native-timing", action="store_true")
    parser.add_argument("--layer-timing", action="store_true")
    parser.add_argument("--windows-residency", action="store_true",
                        help="Record external WDDM adapter gauges outside the timed interval")
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 1:
        parser.error("steps must exceed warmup >= 1")
    if args.output.exists():
        parser.error("Refusing to replace an existing measurement")
    # Do not silently inherit experimental switches from an interactive shell.
    removed = sorted(k for k in os.environ if k.startswith("NSOS_") and
                     k not in ("NSOS_HIP_ROOT", "NSOS_CUDA_ROOT", "NSOS_GPU_DEVICE"))
    for key in removed:
        del os.environ[key]
    environment = profile_environment(args.profile, args.native_timing, args.layer_timing)
    os.environ.update(environment)
    residency_before = windows_adapter_residency() if args.windows_residency else None
    nsos = core.load_nsos(args.build_dir)
    if not nsos.fast_gpu_supported():
        raise RuntimeError("A native GPU is required; no CPU fallback permitted")
    nsos.set_seed(args.seed)
    nsos.set_deterministic_reductions(True)
    nsos.set_strict_gpu_execution(True)
    precision = "bf16" if "bf16" in args.profile else "fp32"
    nsos.set_matmul_precision(precision)
    preset = dict(core.PRESETS["dryrun"], layers=16, d_model=768, seq_len=512,
                  use_gradient_checkpointing=args.profile == "bf16-checkpoint")
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(args.tokenizer.resolve()))
    tokenizer.add_special_tokens(core.SPECIAL_TOKENS)
    config = core.build_model_config(nsos, preset, int(tokenizer.vocab_size), nsos.Device.GPU)
    tokens = array("H")
    with args.shard.open("rb") as handle:
        tokens.frombytes(handle.read((args.steps * 512 + 1) * 2))
    if len(tokens) < args.steps * 512 + 1 or max(tokens) >= config.vocab_size:
        raise RuntimeError("Insufficient or incompatible token data")
    model = nsos.JambaModel(config, nsos.Device.GPU)
    model.to(nsos.Device.GPU)
    trainer = nsos.Trainer(model, 0.0003)
    trainer.weight_decay = 0.1
    trainer.max_grad_norm = 1.0
    trainer.warmup_steps = 500
    trainer.total_training_steps = 24000
    trainer.min_learning_rate_scale = 0.1
    trainer.first_token_loss_scale = 1.0
    trainer.eos_loss_scale = 0.5
    trainer.repetition_unlikelihood_scale = 0.0
    eos = list(tokenizer.encode(core.EOS_TOKEN))
    if len(eos) != 1:
        raise RuntimeError("Tokenizer EOS must be a single token")
    trainer.eos_token_id = eos[0]
    scheduler = trainer.phase_scheduler
    scheduler.progressive_qat_enabled = False
    scheduler.ternary_regularization = 0.0
    trainer.phase_scheduler = scheduler
    initial_weights = model_weight_fingerprint(model)
    samples = []
    for step in range(args.steps):
        if step == args.warmup:
            before = core.runtime_resource_snapshot(nsos, model, trainer)
        begin = step * 512
        # train_step returns a host loss after optimizer commit: wall time includes
        # GPU completion; no extra device synchronization is injected here.
        started = time.perf_counter()
        loss = float(trainer.train_step(list(tokens[begin:begin+512]),
                                        list(tokens[begin+1:begin+513])))
        elapsed = time.perf_counter() - started
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step+1}")
        sample = {"step": step+1, "loss": loss, "seconds": elapsed, "tokens_per_second": 512/elapsed}
        if args.native_timing:
            sample["native_ms"] = {k: float(getattr(trainer.last_step_telemetry, k)) for k in TIMING_FIELDS}
        samples.append(sample)
        if (step+1) % 10 == 0:
            print(f"[{args.profile}] step={step+1} loss={loss:.6f} tok/s={512/elapsed:.1f}", flush=True)
    after = core.runtime_resource_snapshot(nsos, model, trainer)
    final_weights = model_weight_fingerprint(model)
    residency_after = windows_adapter_residency() if args.windows_residency else None
    measured = samples[args.warmup:]
    result = {
        "created_at": core.utc_now(), "profile": args.profile, "precision": precision,
        "purpose": "engine-only; no production checkpoint, no language-quality certification",
        "seed": args.seed, "steps": args.steps, "warmup": args.warmup,
        "environment": environment, "cleared_inherited_switches": removed,
        "binary": str(Path(nsos.__file__).resolve()), "binary_sha256": core.sha256_file(Path(nsos.__file__)),
        "initial_weights": initial_weights, "final_weights": final_weights,
        "windows_residency_before": residency_before, "windows_residency_after": residency_after,
        "script_sha256": core.sha256_file(Path(__file__)),
        "core_script_sha256": core.sha256_file(Path(core.__file__)),
        "shard": str(args.shard.resolve()), "shard_sha256": core.sha256_file(args.shard),
        "tokenizer_sha256": core.sha256_file(args.tokenizer), "devices": nsos.gpu_devices(),
        "selected_device": nsos.selected_gpu_device(), "model": core.parameter_inventory(model),
        "config": core.model_config_dict(config), "resources_before": before, "resources_after": after,
        "transfers_per_step": {k:v/len(measured) for k,v in numeric_delta(after["transfers"], before["transfers"]).items()},
        "throughput_p50": statistics.median(s["tokens_per_second"] for s in measured),
        "throughput_aggregate": len(measured)*512/sum(s["seconds"] for s in measured),
        "samples": samples, "instrumented": args.native_timing or args.layer_timing,
    }
    if args.native_timing:
        result["native_mean_ms"] = {k:statistics.fmean(s["native_ms"][k] for s in measured) for k in TIMING_FIELDS}
    core.atomic_write_json(args.output.resolve(), result)
    print(f"[result] p50={result['throughput_p50']:.2f} tok/s -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
