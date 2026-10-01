"""Pure configuration policies shared by the Oxta Contábil benchmarks.

This module intentionally depends only on the Python standard library so its
resource decisions can be unit-tested without loading NSOS, pandas, a GPU
runtime, or downloaded datasets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


MODEL_LAYERS = 4
MODEL_WIDTH = 128
MAMBA_EXPAND = 2
MAMBA_STATE = 64
MAMBA_HEAD_DIM = 64
MAMBA_GROUPS = 1
MAMBA_CONV_KERNEL = 4
REDUCED_CONV_MAX_KERNEL = 16

# History is only one part of training residency. Auto-checkpoint when the
# worst-case retained SSD histories alone would consume more than 20% of VRAM;
# otherwise retaining them avoids a complete Mamba recomputation.
CHECKPOINT_HISTORY_VRAM_FRACTION = 0.20


def configured_test_inventory(build_dir: Path) -> dict[str, Any]:
    """Read and validate the exact CTest graph emitted by CMake."""

    path = build_dir.resolve() / "nsos_test_inventory.txt"
    if not path.is_file():
        raise RuntimeError(
            f"configured CTest inventory is missing: {path}"
        )
    payload = path.read_bytes()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "configured CTest inventory is not valid UTF-8"
        ) from error
    metadata: dict[str, str] = {}
    sections: dict[str, list[str]] = {
        "tests": [],
        "quarantined": [],
        "gpu_only_sources": [],
        "oxtamem_only_sources": [],
    }
    section_names = set(sections)
    begin_markers = {f"{name}_begin": name for name in section_names}
    end_markers = {f"{name}_end": name for name in section_names}
    seen_sections: set[str] = set()
    active_section: str | None = None
    for line in lines:
        if line in begin_markers:
            if active_section is not None:
                raise RuntimeError(
                    "configured CTest inventory has nested sections"
                )
            active_section = begin_markers[line]
            if active_section in seen_sections:
                raise RuntimeError(
                    "configured CTest inventory repeats a section"
                )
            seen_sections.add(active_section)
            continue
        if line in end_markers:
            expected = end_markers[line]
            if active_section != expected:
                raise RuntimeError(
                    "configured CTest inventory has malformed sections"
                )
            active_section = None
            continue
        if active_section is not None:
            if not line or line in sections[active_section]:
                raise RuntimeError(
                    "configured CTest inventory has an empty or duplicate "
                    "entry"
                )
            sections[active_section].append(line)
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in metadata:
            raise RuntimeError(
                "configured CTest inventory has malformed metadata"
            )
        metadata[key] = value
    if active_section is not None:
        raise RuntimeError(
            "configured CTest inventory has an unterminated section"
        )
    if metadata.get("format") != "nsos-ctest-inventory-v1":
        raise RuntimeError("unsupported configured CTest inventory format")
    if not {"tests", "quarantined"}.issubset(seen_sections):
        raise RuntimeError(
            "configured CTest inventory is missing required sections"
        )
    try:
        test_count = int(metadata.get("test_count", "-1"))
        quarantined_count = int(
            metadata.get("quarantined_count", "-1")
        )
    except ValueError as error:
        raise RuntimeError(
            "configured CTest inventory has a non-integer count"
        ) from error
    if test_count != len(sections["tests"]):
        raise RuntimeError("configured CTest inventory count mismatch")
    if quarantined_count != len(sections["quarantined"]):
        raise RuntimeError(
            "configured CTest quarantine count mismatch"
        )
    optional_counts = {
        "gpu_only_sources": "gpu_only_source_count",
        "oxtamem_only_sources": "oxtamem_only_source_count",
    }
    for section, count_key in optional_counts.items():
        section_present = section in seen_sections
        count_present = count_key in metadata
        if section_present != count_present:
            raise RuntimeError(
                "configured CTest inventory has an incomplete optional "
                f"section: {section}"
            )
        if section_present:
            try:
                count = int(metadata[count_key])
            except ValueError as error:
                raise RuntimeError(
                    "configured CTest inventory has a non-integer count"
                ) from error
            if count != len(sections[section]):
                raise RuntimeError(
                    f"configured CTest inventory {section} count mismatch"
                )
    for section, entries in sections.items():
        if entries != sorted(entries):
            raise RuntimeError(
                "configured CTest inventory section is not canonicalized: "
                f"{section}"
            )
    return {
        **metadata,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **sections,
    }


def estimate_faithful_backward_atomics(
    batch_size: int,
    sequence: int,
    warp_size: int,
) -> dict[str, Any]:
    """Estimate contended global atomics for one faithful training step."""

    if batch_size <= 0 or sequence <= 0 or warp_size <= 0:
        raise ValueError(
            "atomic estimate requires positive batch/sequence/warp size"
        )
    inner = MODEL_WIDTH * MAMBA_EXPAND
    if inner % MAMBA_HEAD_DIM:
        raise ValueError("Mamba inner width is not divisible by head width")
    heads = inner // MAMBA_HEAD_DIM
    group_state = MAMBA_GROUPS * MAMBA_STATE
    conv_dim = inner + 2 * group_state
    channel_count = batch_size * heads * MAMBA_HEAD_DIM

    # Historical scan: B/C, dt, A and D each issued per-channel atomics inside
    # the timestep loop.
    old_bc = (
        2
        * batch_size
        * sequence
        * heads
        * MAMBA_HEAD_DIM
        * MAMBA_STATE
    )
    old_per_head = batch_size * sequence * heads * MAMBA_HEAD_DIM
    old_scan = old_bc + 3 * old_per_head

    if MAMBA_HEAD_DIM % warp_size == 0:
        channel_partials = MAMBA_HEAD_DIM // warp_size
        new_bc = (
            2
            * batch_size
            * sequence
            * heads
            * channel_partials
            * MAMBA_STATE
        )
        new_dt = batch_size * sequence * heads * channel_partials
        new_a_d = 2 * batch_size * heads * channel_partials
        scan_strategy = "warp_aggregated"
    else:
        new_bc = old_bc
        new_dt = old_per_head
        # A/D are still accumulated across the complete sequence in registers.
        new_a_d = 2 * channel_count
        scan_strategy = "scalar_atomic_fallback"
    new_scan = new_bc + new_dt + new_a_d

    kernel = MAMBA_CONV_KERNEL
    valid_taps_per_sequence = (
        kernel * sequence - kernel * (kernel - 1) // 2
        if sequence >= kernel
        else sequence * (sequence + 1) // 2
    )
    old_conv = (
        2 * batch_size * conv_dim * valid_taps_per_sequence
    )
    if kernel <= REDUCED_CONV_MAX_KERNEL:
        new_conv = 0
        conv_strategy = "exclusive_input_block_reduction"
    else:
        new_conv = old_conv
        conv_strategy = "generic_atomic_fallback"

    old_per_layer = old_scan + old_conv
    new_per_layer = new_scan + new_conv
    old_all_layers = old_per_layer * MODEL_LAYERS
    new_all_layers = new_per_layer * MODEL_LAYERS
    return {
        "model_layers": MODEL_LAYERS,
        "heads": heads,
        "head_dim": MAMBA_HEAD_DIM,
        "state": MAMBA_STATE,
        "conv_kernel": kernel,
        "warp_size": warp_size,
        "scan_strategy": scan_strategy,
        "conv_strategy": conv_strategy,
        "historical_scan_atomics_per_layer": old_scan,
        "optimized_scan_atomics_per_layer": new_scan,
        "historical_conv_atomics_per_layer": old_conv,
        "optimized_conv_atomics_per_layer": new_conv,
        "historical_total_atomics_per_layer": old_per_layer,
        "optimized_total_atomics_per_layer": new_per_layer,
        "historical_total_atomics_all_layers": old_all_layers,
        "optimized_total_atomics_all_layers": new_all_layers,
        "estimated_reduction_ratio": (
            old_all_layers / new_all_layers
            if new_all_layers
            else None
        ),
    }


def resolve_gradient_checkpointing(
    requested: str,
    batch_size: int,
    max_sequence: int,
    devices: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve a fail-safe, auditable SSD-history retention policy."""

    if requested not in {"auto", "on", "off"}:
        raise ValueError("invalid gradient-checkpointing policy")
    if batch_size <= 0 or max_sequence <= 0:
        raise ValueError(
            "gradient-checkpointing policy requires positive batch/sequence"
        )
    history_elements = (
        MODEL_LAYERS
        * batch_size
        * max_sequence
        * (MODEL_WIDTH * MAMBA_EXPAND)
        * MAMBA_STATE
    )
    history_bytes = history_elements * 4  # FP32 state history
    eligible_devices = [
        device
        for device in devices
        if bool(device.get("compiled", False))
        and not bool(device.get("integrated", False))
        and int(device.get("total_memory", 0)) > 0
    ]
    selected = (
        max(eligible_devices, key=lambda device: int(device["total_memory"]))
        if eligible_devices
        else None
    )
    total_memory = (
        int(selected["total_memory"]) if selected is not None else None
    )
    reported_warp_size = (
        int(selected.get("warp_size", 0)) if selected is not None else 0
    )
    warp_size = reported_warp_size if reported_warp_size > 0 else 0
    threshold_bytes = (
        int(total_memory * CHECKPOINT_HISTORY_VRAM_FRACTION)
        if total_memory is not None
        else None
    )
    if requested == "on":
        enabled = True
        reason = "forced_on"
    elif requested == "off":
        enabled = False
        reason = "forced_off"
    elif threshold_bytes is None:
        # Unknown capacity must fail toward lower residency, not toward an OOM.
        enabled = True
        reason = "auto_unknown_vram_fail_safe"
    else:
        enabled = history_bytes > threshold_bytes
        reason = (
            "auto_history_exceeds_budget"
            if enabled
            else "auto_history_within_budget"
        )
    return {
        "requested": requested,
        "enabled": enabled,
        "reason": reason,
        "estimated_history_bytes": history_bytes,
        "selected_device_total_memory": total_memory,
        "selected_device_warp_size": warp_size or None,
        "faithful_backward_atomic_estimate": (
            estimate_faithful_backward_atomics(
                batch_size, max_sequence, warp_size
            )
            if warp_size > 0
            else None
        ),
        "history_vram_fraction_limit":
            CHECKPOINT_HISTORY_VRAM_FRACTION,
        "history_budget_bytes": threshold_bytes,
        "model_layers": MODEL_LAYERS,
        "model_width": MODEL_WIDTH,
        "mamba_expand": MAMBA_EXPAND,
        "mamba_state": MAMBA_STATE,
        "batch_size": batch_size,
        "max_sequence": max_sequence,
    }


def validate_training_runtime_contract(
    telemetry: dict[str, Any],
    transfers: dict[str, Any],
    pool: dict[str, Any],
    arm: str,
    steps: int,
    checkpoint_enabled: bool,
    deterministic_reductions: bool = False,
) -> dict[str, Any]:
    """Fail closed when training leaves the declared resident GPU path.

    The ordinary fused optimizer permits two 32-bit device-to-host control
    scalars per step: the reported batch loss and its finite/update gate.
    Deterministic training additionally permits an input finite gate (32-bit)
    plus exactly one ordered global gradient norm (64-bit). The explicit QAT
    arm may additionally report one aggregate ternary-regularization objective
    per step; per-parameter gradients and per-layer scales never qualify. A
    per-sample/per-layer loss loop, tensor staging, global device fence, or
    hidden host fallback exceeds that byte-accurate budget.
    """

    if steps <= 0:
        raise ValueError("runtime contract requires a positive step count")
    mamba_layers = len(telemetry.get("mamba_layers", []))
    expected_mamba_calls = steps * mamba_layers
    expected_recomputes = (
        expected_mamba_calls if checkpoint_enabled else 0
    )
    expected_forwards = expected_mamba_calls + expected_recomputes
    qat_objective_scalars_per_step = 1 if arm == "hybrid_qat" else 0
    deterministic_f32_scalars_per_step = (
        1 if deterministic_reductions else 0
    )
    deterministic_f64_scalars_per_step = (
        1 if deterministic_reductions else 0
    )
    control_scalars_per_step = (
        2
        + deterministic_f32_scalars_per_step
        + deterministic_f64_scalars_per_step
        + qat_objective_scalars_per_step
    )
    allowed_d2h_calls = control_scalars_per_step * steps
    allowed_d2h_bytes = steps * (
        2 * 4
        + deterministic_f32_scalars_per_step * 4
        + deterministic_f64_scalars_per_step * 8
        + qat_objective_scalars_per_step * 4
    )
    checks: dict[str, Any] = {
        "arm": arm,
        "mamba_layers": mamba_layers,
        "expected_mamba_backward_calls": expected_mamba_calls,
        "expected_mamba_forward_calls": expected_forwards,
        "expected_selective_history_recomputes": expected_recomputes,
        "allowed_d2h_control_scalars": {
            "loss_f32_per_step": 1,
            "optimizer_update_gate_i32_per_step": 1,
            "deterministic_input_finite_gate_i32_per_step":
                deterministic_f32_scalars_per_step,
            "deterministic_grad_norm_f64_per_step":
                deterministic_f64_scalars_per_step,
            "qat_regularization_f32_per_step":
                qat_objective_scalars_per_step,
            "max_calls": allowed_d2h_calls,
            "max_bytes": allowed_d2h_bytes,
        },
        "bounded_d2h_control_scalars": (
            0 <= int(transfers.get("d2h_calls", -1))
            <= allowed_d2h_calls
            and 0 <= int(transfers.get("d2h_bytes", -1))
            <= allowed_d2h_bytes
        ),
        "zero_host_fallbacks": (
            int(telemetry.get("mamba_fast_path_fallbacks", 0)) == 0
            and int(
                telemetry.get(
                    "faithful_forward_host_fallbacks", 0
                )
            )
            == 0
            and int(
                telemetry.get(
                    "faithful_backward_host_fallbacks", 0
                )
            )
            == 0
        ),
        "zero_device_synchronizations": (
            int(transfers.get("device_synchronizations", 0)) == 0
        ),
        "bounded_control_stream_synchronizations": (
            0 <= int(transfers.get("stream_synchronizations", -1))
            <= allowed_d2h_calls
        ),
        "device_memory_only": (
            int(pool.get("managed_cached_bytes", 0)) == 0
            and int(pool.get("managed_live_bytes", 0)) == 0
            and int(pool.get("managed_pressure_probes", 0)) == 0
            and int(pool.get("managed_advice_failures", 0)) == 0
            and int(pool.get("cross_stream_domain_frees", 0)) == 0
        ),
        "pool_integrity": (
            int(pool.get("retained_release_bytes", 0)) == 0
            and int(pool.get("retained_release_blocks", 0)) == 0
            and int(pool.get("release_failures", 0)) == 0
            and int(
                pool.get("unknown_deallocation_attempts", 0)
            )
            == 0
            and int(pool.get("capture_contract_violations", 0)) == 0
        ),
    }
    if mamba_layers:
        checks.update(
            {
                "forward_count_matches": (
                    int(
                        telemetry.get(
                            "faithful_forward_gpu_calls", -1
                        )
                    )
                    == expected_forwards
                ),
                "backward_count_matches": (
                    int(
                        telemetry.get(
                            "faithful_backward_gpu_calls", -1
                        )
                    )
                    == expected_mamba_calls
                ),
                "checkpoint_strategy_matches": (
                    int(
                        telemetry.get(
                            "faithful_recompute_forwards", -1
                        )
                    )
                    == expected_recomputes
                    and int(
                        telemetry.get(
                            "faithful_selective_history_recomputes", -1
                        )
                    )
                    == expected_recomputes
                    and int(
                        telemetry.get(
                            "faithful_full_block_recompute_forwards", -1
                        )
                    )
                    == 0
                ),
                "audited_scan_reduction_only": (
                    int(
                        telemetry.get(
                            "faithful_scalar_atomic_backward_calls", -1
                        )
                    )
                    == 0
                    and (
                        (
                            int(
                                telemetry.get(
                                    "faithful_warp_aggregated_backward_calls",
                                    -1,
                                )
                            )
                            == expected_mamba_calls
                            and int(
                                telemetry.get(
                                    "faithful_deterministic_backward_calls",
                                    -1,
                                )
                            )
                            == 0
                        )
                        or (
                            int(
                                telemetry.get(
                                    "faithful_deterministic_backward_calls",
                                    -1,
                                )
                            )
                            == expected_mamba_calls
                            and int(
                                telemetry.get(
                                    "faithful_warp_aggregated_backward_calls",
                                    -1,
                                )
                            )
                            == 0
                        )
                    )
                ),
                "reduced_convolution_only": (
                    int(
                        telemetry.get(
                            "faithful_reduced_conv_backward_calls", -1
                        )
                    )
                    == expected_mamba_calls
                    and int(
                        telemetry.get(
                            "faithful_generic_atomic_conv_backward_calls", -1
                        )
                    )
                    == 0
                ),
                "grouped_projection_counts_match": (
                    int(
                        telemetry.get(
                            "faithful_grouped_projection_forward_calls", -1
                        )
                    )
                    == expected_mamba_calls
                    and int(
                        telemetry.get(
                            "faithful_grouped_projection_backward_calls", -1
                        )
                    )
                    == expected_mamba_calls
                ),
            }
        )
    failed = [
        name
        for name, passed in checks.items()
        if isinstance(passed, bool) and not passed
    ]
    checks["passed"] = not failed
    checks["failed_checks"] = failed
    if failed:
        observed = {
            "d2h_calls": int(transfers.get("d2h_calls", -1)),
            "d2h_bytes": int(transfers.get("d2h_bytes", -1)),
            "device_synchronizations": int(
                transfers.get("device_synchronizations", -1)
            ),
            "stream_synchronizations": int(
                transfers.get("stream_synchronizations", -1)
            ),
        }
        raise RuntimeError(
            f"{arm} GPU runtime contract failed: {', '.join(failed)}; "
            f"observed={observed}; "
            f"allowed_d2h_calls={allowed_d2h_calls}; "
            f"allowed_d2h_bytes={allowed_d2h_bytes}"
        )
    return checks
