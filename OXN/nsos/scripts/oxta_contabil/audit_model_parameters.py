"""Emit a deterministic layer and parameter manifest for NSOS product profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROFILES = ("mamba", "hybrid_parallel", "hybrid_parallel_last_mamba")
MANIFEST_SEED = 20260728


def tensor_sha256_f32_le(values: np.ndarray, shape: list[int]) -> str:
    """Match LayerAudit's NSOS-TENSOR-F32-LE-v1 canonical digest."""
    digest = hashlib.sha256()
    digest.update(b"NSOS-TENSOR-F32-LE-v1\x00")
    digest.update(struct.pack("<Q", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=root / "build-codex-hip")
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "artifacts"
        / "oxta_contabil_amd"
        / "audit"
        / "product_parameter_manifest.json",
    )
    return parser.parse_args()


def load_nsos(build_dir: Path):
    build_dir = build_dir.resolve()
    if os.name == "nt":
        runtime_dirs = (
            Path(r"C:\TheRock\build\bin"),
            Path(r"C:\TheRock\build\lib\llvm\bin"),
            build_dir,
        )
        load_nsos.dll_handles = [
            os.add_dll_directory(str(path)) for path in runtime_dirs if path.is_dir()
        ]
    sys.path.insert(0, str(build_dir))
    import nsos_ext as nsos  # type: ignore

    return nsos


load_nsos.dll_handles = []


def base_config(nsos):
    config = nsos.ModelConfig()
    config.architecture_schema_version = 2
    config.num_layers = 4
    config.d_model = 128
    config.vocab_size = 25
    config.n_heads = 4
    config.n_kv_heads = 2
    config.max_context_tokens = 128
    config.default_batch_size = 32
    config.use_cuda = False
    config.dropout = 0.0
    config.use_moe = False
    config.use_kan = False
    config.use_ttt = False
    config.use_chrass = False
    config.use_slender_embedding = False
    config.mamba2_faithful = True
    config.mamba_state_expansion = True
    config.mamba_expand = 2
    config.mamba_head_dim = 64
    config.mamba_d_state = 64
    config.mamba_n_groups = 1
    config.use_flash_attn = False
    config.use_exact_attention_training = True
    config.use_gradient_checkpointing = True
    config.force_mamba_last_layer = False
    config.faithful_attention_linears = True
    config.hybrid_mamba_gate_init = 1.0
    config.hybrid_attention_gate_init = 0.01
    config.hybrid_ffn_gate_init = 0.01
    return config


def classify_parameter(name: str, base_name: str) -> dict[str, Any]:
    parts = name.split(".")
    layer_index = None
    component = parts[0]
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        layer_index = int(parts[1])
        component = parts[2]
    if base_name in {"flat_alpha", "flat_beta"}:
        role = "legacy_fixed_identity"
    elif base_name in {"A", "D"}:
        role = "ssm_state"
    elif "ssa_wsel" in name:
        role = "attention_selector"
    elif "layerscale" in name:
        role = "residual_scale"
    elif base_name.endswith("gate") or name.endswith(".gate"):
        role = "branch_gate"
    elif "norm" in name:
        role = "normalization"
    elif base_name.endswith("bias") or base_name == "bias":
        role = "bias"
    elif base_name.endswith("weight") or base_name == "weight":
        role = "weight"
    else:
        raise RuntimeError(
            f"unclassified product parameter: name={name!r} "
            f"base_name={base_name!r}"
        )
    return {
        "layer_index": layer_index,
        "component": component,
        "role": role,
        "trainable": role != "legacy_fixed_identity",
    }


def effective_topology(
    config, *, parallel_hybrid: bool
) -> list[dict[str, Any]]:
    topology = []
    for layer_index in range(config.num_layers):
        one_based = layer_index + 1
        period = max(int(config.attention_period), 1)
        slot = min(max(int(config.attention_slot), 0), period - 1)
        attention = ((one_based - 1) % period) == slot
        guard_applied = bool(
            attention
            and config.mamba2_faithful
            and one_based == config.num_layers
            and config.force_mamba_last_layer
        )
        if guard_applied or not attention:
            effective_type = "mamba"
        elif parallel_hybrid:
            effective_type = "mamba+attention"
        else:
            effective_type = "attention_replacement"
        topology.append(
            {
                "index": layer_index,
                "one_based": one_based,
                "scheduled_attention": attention,
                "last_layer_guard_applied": guard_applied,
                "effective_type": effective_type,
            }
        )
    return topology


def build_profile(nsos, profile: str) -> dict[str, Any]:
    nsos.set_seed(MANIFEST_SEED)
    config = base_config(nsos)
    if profile == "mamba":
        config.attention_period = 64
        config.attention_slot = 63
        config.force_mamba_last_layer = False
    else:
        config.attention_period = 2
        config.attention_slot = 1
        config.force_mamba_last_layer = (
            profile == "hybrid_parallel_last_mamba"
        )
    config.hybrid_composition = nsos.HybridComposition.PARALLEL_GATED
    config.faithful_attention_linears = True

    model = nsos.JambaModel(config, nsos.Device.CPU)
    parameters = []
    names: set[str] = set()
    for index, parameter in enumerate(model.parameters()):
        if not parameter.name or parameter.name in names:
            raise RuntimeError(
                f"parameter registry has missing/duplicate name: {parameter.name!r}"
            )
        names.add(parameter.name)
        classification = classify_parameter(parameter.name, parameter.base_name)
        declared_trainable = bool(parameter.trainable)
        if declared_trainable != bool(classification["trainable"]):
            raise RuntimeError(
                "parameter trainability contract mismatch: "
                f"name={parameter.name!r} "
                f"declared={declared_trainable} "
                f"classified={classification['trainable']}"
            )
        shape = list(parameter.data.shape)
        values = np.asarray(parameter.data.numpy(), dtype="<f4").reshape(-1)
        finite_values = values[np.isfinite(values)]
        value_sha256 = tensor_sha256_f32_le(values, shape)
        parameters.append(
            {
                "registry_index": index,
                "name": parameter.name,
                "base_name": parameter.base_name,
                "shape": shape,
                "elements": int(parameter.data.size),
                "fp32_bytes": int(parameter.data.size) * 4,
                "value_sha256_f32_le": value_sha256,
                "statistics": {
                    "finite": bool(np.isfinite(values).all()),
                    "nan_count": int(np.isnan(values).sum()),
                    "inf_count": int(np.isinf(values).sum()),
                    "zero_count": int(np.count_nonzero(values == 0.0)),
                    "min": (
                        float(finite_values.min())
                        if finite_values.size
                        else None
                    ),
                    "max": (
                        float(finite_values.max())
                        if finite_values.size
                        else None
                    ),
                    "mean": (
                        float(finite_values.astype(np.float64).mean())
                        if finite_values.size
                        else None
                    ),
                    "l2_norm": (
                        float(
                            np.linalg.norm(
                                finite_values.astype(np.float64)
                            )
                        )
                        if finite_values.size
                        else None
                    ),
                },
                **classification,
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

    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "elements": 0, "trainable_elements": 0}
    )
    for parameter in parameters:
        layer = parameter["layer_index"]
        group = (
            f"layer_{layer}.{parameter['component']}"
            if layer is not None
            else str(parameter["component"])
        )
        groups[group]["tensors"] += 1
        groups[group]["elements"] += parameter["elements"]
        if parameter["trainable"]:
            groups[group]["trainable_elements"] += parameter["elements"]

    topology = effective_topology(config, parallel_hybrid=True)
    layer_names: dict[int, list[str]] = defaultdict(list)
    for parameter in parameters:
        if parameter["layer_index"] is not None:
            layer_names[int(parameter["layer_index"])].append(
                str(parameter["name"])
            )
    expected_layers = set(range(int(config.num_layers)))
    if set(layer_names) != expected_layers:
        raise RuntimeError(
            "parameter registry layer coverage mismatch: "
            f"expected={sorted(expected_layers)} "
            f"actual={sorted(layer_names)}"
        )
    for layer in topology:
        index = int(layer["index"])
        names = layer_names[index]
        has_mamba = any(
            name.startswith(f"layers.{index}.mamba.")
            for name in names
        )
        has_attention = any(
            name.startswith(f"layers.{index}.attn.")
            for name in names
        )
        expected_type = layer["effective_type"]
        if not has_mamba:
            raise RuntimeError(
                f"product layer {index} lacks its Mamba parameters"
            )
        if expected_type == "mamba+attention":
            required = {
                f"layers.{index}.mamba.gate",
                f"layers.{index}.attn.gate",
                f"layers.{index}.ffn.gate",
                f"layers.{index}.mamba.pre_norm.weight",
                f"layers.{index}.attn.pre_norm.weight",
                f"layers.{index}.ffn.pre_norm.weight",
            }
            missing = sorted(required.difference(names))
            if not has_attention or missing:
                raise RuntimeError(
                    f"hybrid layer {index} registry is incomplete; "
                    f"missing={missing}"
                )
        elif has_attention:
            raise RuntimeError(
                f"Mamba-only product layer {index} unexpectedly exposes "
                "Attention parameters"
            )

    fingerprint_payload = [
        {
            "name": row["name"],
            "shape": row["shape"],
            "trainable": row["trainable"],
        }
        for row in parameters
    ]
    fingerprint_payload.extend(
        {
            "alias": alias["alias"],
            "canonical": alias["canonical"],
        }
        for alias in aliases
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "profile": profile,
        "config": {
            "num_layers": config.num_layers,
            "d_model": config.d_model,
            "vocab_size": config.vocab_size,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "mamba2_faithful": config.mamba2_faithful,
            "mamba_expand": config.mamba_expand,
            "mamba_head_dim": config.mamba_head_dim,
            "mamba_d_state": config.mamba_d_state,
            "mamba_n_groups": config.mamba_n_groups,
            "use_gradient_checkpointing":
                config.use_gradient_checkpointing,
            "attention_period": config.attention_period,
            "attention_slot": config.attention_slot,
            "hybrid_composition": "parallel_gated",
            "force_mamba_last_layer": config.force_mamba_last_layer,
            "faithful_attention_linears":
                config.faithful_attention_linears,
            "hybrid_mamba_gate_init": config.hybrid_mamba_gate_init,
            "hybrid_attention_gate_init":
                config.hybrid_attention_gate_init,
            "hybrid_ffn_gate_init": config.hybrid_ffn_gate_init,
        },
        "topology": topology,
        "summary": {
            "registry_tensors": len(parameters),
            "logical_parameter_paths":
                len(parameters) + len(aliases),
            "trainable_tensors": sum(row["trainable"] for row in parameters),
            "fixed_legacy_tensors": sum(not row["trainable"] for row in parameters),
            "registry_elements": sum(row["elements"] for row in parameters),
            "trainable_elements": sum(
                row["elements"] for row in parameters if row["trainable"]
            ),
            "fixed_legacy_elements": sum(
                row["elements"] for row in parameters if not row["trainable"]
            ),
            "fp32_registry_bytes": sum(row["fp32_bytes"] for row in parameters),
            "parameter_manifest_sha256": fingerprint,
        },
        "groups": dict(sorted(groups.items())),
        "aliases": aliases,
        "parameters": parameters,
    }


def main() -> int:
    args = parse_args()
    nsos = load_nsos(args.build_dir)
    profiles = [build_profile(nsos, profile) for profile in PROFILES]
    report = {
        "schema_version": 3,
        "purpose": "deterministic topology and parameter registry audit",
        "initialization_seed": MANIFEST_SEED,
        "value_hash_format": "sha256/nsos-tensor-f32-le-v1",
        "profiles": profiles,
        "known_registry_contract": {
            "legacy_fixed_parameters": ["flat_alpha", "flat_beta"],
            "note": (
                "registry_elements includes fixed checkpoint-compatibility buffers; "
                "trainable_elements excludes them"
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=output.name + ".tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"REPORT {output}")
    for profile in profiles:
        summary = profile["summary"]
        topology = "-".join(
            layer["effective_type"][0].upper() for layer in profile["topology"]
        )
        print(
            f"{profile['profile']}: topology={topology} "
            f"registry={summary['registry_elements']} "
            f"trainable={summary['trainable_elements']} "
            f"tensors={summary['registry_tensors']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
