from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "require_zero_nan_inf": True,
    "min_layer_coverage": 1,
    "router_entropy_min": 0.0,
    "router_entropy_max": 0.0,
    "max_reload_drift": 1.0e-5,
}


def _empty_phase_summary(name: str) -> Dict[str, Any]:
    return {
        "phase": name,
        "records": 0,
        "forward_records": 0,
        "backward_records": 0,
        "router_records": 0,
        "token_contexts": 0,
        "training_steps": 0,
        "layers_seen": set(),
        "total_nan": 0,
        "total_inf": 0,
        "max_latency_ms": 0.0,
        "max_l2_norm": 0.0,
        "max_grad_l2_norm": 0.0,
        "router_entropy_values": [],
        "router_entropy_stats": None,
        "router_num_experts": [],
        "loss_values": [],
        "grad_l2_norm_values": [],
        "truncated_contexts": 0,
    }


def _stats(values: Iterable[float]) -> Dict[str, float]:
    clean = [float(value) for value in values]
    if not clean:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "last": 0.0}
    return {
        "min": min(clean),
        "max": max(clean),
        "mean": sum(clean) / len(clean),
        "last": clean[-1],
    }


def _tensor_nan_inf(record: Dict[str, Any], side: str, key: str) -> int:
    tensor = record.get(side, {})
    return int(tensor.get(key, 0) or 0)


def _tensor_l2(record: Dict[str, Any], side: str) -> float:
    tensor = record.get(side, {})
    return float(tensor.get("l2_norm", 0.0) or 0.0)


def _as_serializable_phase(phase: Dict[str, Any]) -> Dict[str, Any]:
    entropies = phase["router_entropy_values"]
    experts = phase["router_num_experts"]
    result = dict(phase)
    result["layers_seen"] = sorted(int(layer) for layer in phase["layers_seen"])
    result["layer_coverage"] = len(result["layers_seen"])
    result["router_entropy"] = phase.get("router_entropy_stats") or _stats(entropies)
    result["router_entropy_values"] = [float(value) for value in entropies]
    result["router_num_experts"] = [int(value) for value in experts]
    result["loss"] = _stats(phase["loss_values"])
    result["grad_l2_norm"] = _stats(phase["grad_l2_norm_values"])
    result["loss_values"] = [float(value) for value in phase["loss_values"]]
    result["grad_l2_norm_values"] = [float(value) for value in phase["grad_l2_norm_values"]]
    return result


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def summarize_audit_json(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    phases: Dict[str, Dict[str, Any]] = {}
    totals = {
        "records": 0,
        "forward_records": 0,
        "backward_records": 0,
        "router_records": 0,
        "token_contexts": 0,
        "training_steps": 0,
        "total_nan": 0,
        "total_inf": 0,
        "max_latency_ms": 0.0,
        "max_l2_norm": 0.0,
        "max_grad_l2_norm": 0.0,
    }

    phase_summaries = data.get("phase_summaries", [])
    if phase_summaries:
        for item in phase_summaries:
            phase_name = str(item.get("phase", "default"))
            phase = phases.setdefault(phase_name, _empty_phase_summary(phase_name))
            phase["records"] = int(item.get("records", 0) or 0)
            phase["forward_records"] = int(item.get("forward_records", 0) or 0)
            phase["backward_records"] = int(item.get("backward_records", 0) or 0)
            phase["router_records"] = int(item.get("router_records", 0) or 0)
            phase["token_contexts"] = int(item.get("token_contexts", 0) or 0)
            phase["training_steps"] = int(item.get("training_steps", 0) or 0)
            phase["total_nan"] = int(item.get("total_nan", 0) or 0)
            phase["total_inf"] = int(item.get("total_inf", 0) or 0)
            phase["max_latency_ms"] = float(item.get("max_latency_ms", 0.0) or 0.0)
            phase["max_l2_norm"] = float(item.get("max_l2_norm", 0.0) or 0.0)
            phase["stored_records"] = int(item.get("stored_records", 0) or 0)
            phase["dropped_records"] = int(item.get("dropped_records", 0) or 0)
            phase["truncated_contexts"] = int(item.get("truncated_contexts", 0) or 0)
            phase["layers_seen"] = set(int(layer) for layer in item.get("layers_seen", []))
            entropy_count = int(item.get("router_entropy_count", 0) or 0)
            if entropy_count > 0:
                entropy_min = float(item.get("router_entropy_min", 0.0) or 0.0)
                entropy_max = float(item.get("router_entropy_max", 0.0) or 0.0)
                entropy_mean = float(item.get("router_entropy_mean", 0.0) or 0.0)
                phase["router_entropy_values"] = [
                    entropy_min,
                    entropy_max,
                ]
                phase["router_entropy_stats"] = {
                    "min": entropy_min,
                    "max": entropy_max,
                    "mean": entropy_mean,
                    "last": entropy_mean,
                }
                phase["router_num_experts"] = [
                    int(item.get("router_num_experts_max", 0) or 0)
                ]
            for key in [
                "records",
                "forward_records",
                "backward_records",
                "router_records",
                "token_contexts",
                "training_steps",
                "total_nan",
                "total_inf",
            ]:
                totals[key] += int(phase.get(key, 0))
            totals["max_latency_ms"] = max(
                float(totals["max_latency_ms"]),
                float(phase.get("max_latency_ms", 0.0)),
            )
            totals["max_l2_norm"] = max(
                float(totals["max_l2_norm"]),
                float(phase.get("max_l2_norm", 0.0)),
            )

    for record in ([] if phase_summaries else data.get("records", [])):
        phase_name = str(record.get("phase", "default"))
        phase = phases.setdefault(phase_name, _empty_phase_summary(phase_name))
        pass_name = str(record.get("pass", ""))
        layer_index = int(record.get("layer_index", -1))
        nan_count = (
            _tensor_nan_inf(record, "input", "nan_count")
            + _tensor_nan_inf(record, "output", "nan_count")
        )
        inf_count = (
            _tensor_nan_inf(record, "input", "inf_count")
            + _tensor_nan_inf(record, "output", "inf_count")
        )
        max_l2 = max(_tensor_l2(record, "input"), _tensor_l2(record, "output"))
        grad_l2 = float(record.get("grad_l2_norm", 0.0) or 0.0)
        latency_ms = float(record.get("latency_ms", 0.0) or 0.0)

        phase["records"] += 1
        totals["records"] += 1
        if pass_name == "forward":
            phase["forward_records"] += 1
            totals["forward_records"] += 1
        elif pass_name == "backward":
            phase["backward_records"] += 1
            totals["backward_records"] += 1
        elif pass_name == "router":
            phase["router_records"] += 1
            totals["router_records"] += 1
            router = record.get("router", {})
            phase["router_entropy_values"].append(float(router.get("entropy", 0.0) or 0.0))
            phase["router_num_experts"].append(int(router.get("num_experts", 0) or 0))

        if layer_index >= 0:
            phase["layers_seen"].add(layer_index)
        phase["total_nan"] += nan_count
        phase["total_inf"] += inf_count
        phase["max_latency_ms"] = max(float(phase["max_latency_ms"]), latency_ms)
        phase["max_l2_norm"] = max(float(phase["max_l2_norm"]), max_l2)
        phase["max_grad_l2_norm"] = max(float(phase["max_grad_l2_norm"]), grad_l2)

        totals["total_nan"] += nan_count
        totals["total_inf"] += inf_count
        totals["max_latency_ms"] = max(float(totals["max_latency_ms"]), latency_ms)
        totals["max_l2_norm"] = max(float(totals["max_l2_norm"]), max_l2)
        totals["max_grad_l2_norm"] = max(float(totals["max_grad_l2_norm"]), grad_l2)

    for context in data.get("token_contexts", []):
        phase_name = str(context.get("phase", "default"))
        phase = phases.setdefault(phase_name, _empty_phase_summary(phase_name))
        if not phase_summaries:
            phase["token_contexts"] += 1
            phase["truncated_contexts"] += 1 if bool(context.get("truncated", False)) else 0
            totals["token_contexts"] += 1

    for step in data.get("training_steps", []):
        phase_name = str(step.get("phase", "default"))
        phase = phases.setdefault(phase_name, _empty_phase_summary(phase_name))
        if not phase_summaries:
            phase["training_steps"] += 1
        phase["loss_values"].append(float(step.get("loss", 0.0) or 0.0))
        grad_l2 = float(step.get("grad_l2_norm", 0.0) or 0.0)
        phase["grad_l2_norm_values"].append(grad_l2)
        phase["max_grad_l2_norm"] = max(float(phase["max_grad_l2_norm"]), grad_l2)
        if not phase_summaries:
            totals["training_steps"] += 1
        totals["max_grad_l2_norm"] = max(float(totals["max_grad_l2_norm"]), grad_l2)

    serial_phases = {
        name: _as_serializable_phase(phases[name])
        for name in sorted(phases)
    }
    return {
        "path": str(path),
        "run_id": str(data.get("run_id", "")),
        "totals": totals,
        "phases": serial_phases,
    }


def summarize_metrics_jsonl(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    phase_rows = [row for row in rows if row.get("phase")]
    return {
        "path": str(path),
        "phase_count": len(phase_rows),
        "curves": {
            "ema_loss_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("ema_loss_final", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
            "answer_loss_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("answer_loss", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
            "heldout_loss_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("heldout_loss", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
            "first_token_accuracy_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("first_token_accuracy", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
            "teacher_token_accuracy_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("teacher_token_accuracy", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
            "exact_accuracy_by_phase": [
                {
                    "phase": row.get("phase", ""),
                    "value": float(row.get("exact_accuracy", 0.0) or 0.0),
                }
                for row in phase_rows
            ],
        },
    }


def summarize_run_summary(path: Path) -> Dict[str, Any]:
    summary = load_json(path)
    holdouts = summary.get("official_holdouts", {})
    layer_audit = summary.get("layer_audit", {})
    return {
        "path": str(path),
        "profile": summary.get("profile", ""),
        "requested_profile": summary.get("requested_profile", ""),
        "device": summary.get("device", ""),
        "official_holdouts": holdouts,
        "reload_probe": layer_audit.get("reload_probe", {}),
        "thresholds": layer_audit.get("thresholds", {}),
        "audit_gate": layer_audit.get("gate", {}),
    }


def _threshold_failures(audit: Dict[str, Any],
                        thresholds: Dict[str, Any],
                        reload_probe: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    totals = audit["totals"]
    if thresholds.get("require_zero_nan_inf", True):
        if int(totals.get("total_nan", 0)) != 0:
            failures.append({
                "scope": audit["path"],
                "kind": "nan",
                "observed": int(totals.get("total_nan", 0)),
                "threshold": 0,
            })
        if int(totals.get("total_inf", 0)) != 0:
            failures.append({
                "scope": audit["path"],
                "kind": "inf",
                "observed": int(totals.get("total_inf", 0)),
                "threshold": 0,
            })

    min_layer_coverage = int(thresholds.get("min_layer_coverage", 1) or 0)
    if min_layer_coverage > 0:
        for phase_name, phase in audit["phases"].items():
            if int(phase.get("records", 0)) <= 0:
                continue
            coverage = int(phase.get("layer_coverage", 0))
            if coverage < min_layer_coverage:
                failures.append({
                    "scope": phase_name,
                    "kind": "layer_coverage",
                    "observed": coverage,
                    "threshold": min_layer_coverage,
                })

    router_records = int(totals.get("router_records", 0))
    if router_records > 0:
        entropy_min = float(thresholds.get("router_entropy_min", 0.0) or 0.0)
        entropy_max = float(thresholds.get("router_entropy_max", 0.0) or 0.0)
        for phase_name, phase in audit["phases"].items():
            values = phase.get("router_entropy_values", [])
            if not values:
                continue
            max_experts = max([int(value) for value in phase.get("router_num_experts", [])] or [0])
            effective_max = entropy_max
            if effective_max <= 0.0 and max_experts > 1:
                effective_max = math.log(max_experts, 2.0) + 0.05
            for value in values:
                if float(value) < entropy_min:
                    failures.append({
                        "scope": phase_name,
                        "kind": "router_entropy_low",
                        "observed": float(value),
                        "threshold": entropy_min,
                    })
                if effective_max > 0.0 and float(value) > effective_max:
                    failures.append({
                        "scope": phase_name,
                        "kind": "router_entropy_high",
                        "observed": float(value),
                        "threshold": effective_max,
                    })

    if reload_probe:
        drift = float(reload_probe.get("max_abs_drift", 0.0) or 0.0)
        max_reload_drift = float(thresholds.get("max_reload_drift", 1.0e-5) or 0.0)
        if max_reload_drift >= 0.0 and drift > max_reload_drift:
            failures.append({
                "scope": "reload_probe",
                "kind": "reload_drift",
                "observed": drift,
                "threshold": max_reload_drift,
            })

    return failures


def _build_curves(audits: List[Dict[str, Any]],
                  metrics: List[Dict[str, Any]],
                  run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    audit_curves = []
    for audit in audits:
        audit_curves.append({
            "path": audit["path"],
            "loss_by_audit_phase": [
                {
                    "phase": name,
                    "last": phase["loss"]["last"],
                    "mean": phase["loss"]["mean"],
                }
                for name, phase in audit["phases"].items()
                if phase.get("training_steps", 0) > 0
            ],
            "grad_norm_by_audit_phase": [
                {
                    "phase": name,
                    "last": phase["grad_l2_norm"]["last"],
                    "max": phase["grad_l2_norm"]["max"],
                    "mean": phase["grad_l2_norm"]["mean"],
                }
                for name, phase in audit["phases"].items()
                if phase.get("training_steps", 0) > 0
            ],
            "router_entropy_by_audit_phase": [
                {
                    "phase": name,
                    "min": phase["router_entropy"]["min"],
                    "max": phase["router_entropy"]["max"],
                    "mean": phase["router_entropy"]["mean"],
                }
                for name, phase in audit["phases"].items()
                if phase.get("router_records", 0) > 0
            ],
            "nan_inf_by_audit_phase": [
                {
                    "phase": name,
                    "nan": int(phase.get("total_nan", 0)),
                    "inf": int(phase.get("total_inf", 0)),
                }
                for name, phase in audit["phases"].items()
            ],
        })

    return {
        "audit": audit_curves,
        "metrics": metrics,
        "holdouts": [
            {
                "path": summary["path"],
                "official_holdouts": summary.get("official_holdouts", {}),
                "reload_probe": summary.get("reload_probe", {}),
            }
            for summary in run_summaries
        ],
    }


def _compare_audits(audits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(audits) < 2:
        return []
    baseline = audits[0]
    comparisons: List[Dict[str, Any]] = []
    for current in audits[1:]:
        phase_deltas: Dict[str, Any] = {}
        for phase_name, current_phase in current["phases"].items():
            base_phase = baseline["phases"].get(phase_name)
            if not base_phase:
                phase_deltas[phase_name] = {"status": "new_phase"}
                continue
            phase_deltas[phase_name] = {
                "records_delta": int(current_phase.get("records", 0)) - int(base_phase.get("records", 0)),
                "max_l2_norm_delta": (
                    float(current_phase.get("max_l2_norm", 0.0))
                    - float(base_phase.get("max_l2_norm", 0.0))
                ),
                "max_grad_l2_norm_delta": (
                    float(current_phase.get("max_grad_l2_norm", 0.0))
                    - float(base_phase.get("max_grad_l2_norm", 0.0))
                ),
                "router_entropy_mean_delta": (
                    float(current_phase.get("router_entropy", {}).get("mean", 0.0))
                    - float(base_phase.get("router_entropy", {}).get("mean", 0.0))
                ),
                "nan_delta": int(current_phase.get("total_nan", 0)) - int(base_phase.get("total_nan", 0)),
                "inf_delta": int(current_phase.get("total_inf", 0)) - int(base_phase.get("total_inf", 0)),
            }
        comparisons.append({
            "baseline": baseline["path"],
            "current": current["path"],
            "totals_delta": {
                key: current["totals"].get(key, 0) - baseline["totals"].get(key, 0)
                for key in [
                    "records",
                    "forward_records",
                    "backward_records",
                    "router_records",
                    "token_contexts",
                    "training_steps",
                    "total_nan",
                    "total_inf",
                ]
            },
            "phase_deltas": phase_deltas,
        })
    return comparisons


def build_report(audit_paths: List[Path],
                 metrics_paths: List[Path] | None = None,
                 run_summary_paths: List[Path] | None = None,
                 thresholds: Dict[str, Any] | None = None,
                 run_summaries: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    effective_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update(thresholds)

    audits = [summarize_audit_json(path) for path in audit_paths]
    metrics = [summarize_metrics_jsonl(path) for path in (metrics_paths or [])]
    loaded_run_summaries = [
        summarize_run_summary(path) for path in (run_summary_paths or [])
    ]
    if run_summaries:
        loaded_run_summaries.extend(run_summaries)

    failures: List[Dict[str, Any]] = []
    for index, audit in enumerate(audits):
        reload_probe = None
        if index < len(loaded_run_summaries):
            reload_probe = loaded_run_summaries[index].get("reload_probe", {})
        failures.extend(_threshold_failures(audit, effective_thresholds, reload_probe))

    return {
        "verdict": "pass" if not failures else "fail",
        "thresholds": effective_thresholds,
        "failures": failures,
        "audits": audits,
        "curves": _build_curves(audits, metrics, loaded_run_summaries),
        "comparisons": _compare_audits(audits),
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"verdict={report['verdict']}")
    for audit in report.get("audits", []):
        totals = audit["totals"]
        print(
            "audit="
            f"{audit['path']} records={totals['records']} "
            f"forward={totals['forward_records']} backward={totals['backward_records']} "
            f"router={totals['router_records']} training_steps={totals['training_steps']} "
            f"nan={totals['total_nan']} inf={totals['total_inf']}"
        )
    if report.get("failures"):
        print("failures:")
        for failure in report["failures"]:
            print(
                f"  {failure['kind']} scope={failure['scope']} "
                f"observed={failure['observed']} threshold={failure['threshold']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and compare NSOS layer audit JSONs.")
    parser.add_argument("--audit", type=Path, action="append", required=True, help="Layer audit JSON path.")
    parser.add_argument("--metrics", type=Path, action="append", default=[], help="metrics.jsonl path.")
    parser.add_argument("--run-summary", type=Path, action="append", default=[], help="run_summary.json path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path.")
    parser.add_argument("--min-layer-coverage", type=int, default=1)
    parser.add_argument("--router-entropy-min", type=float, default=0.0)
    parser.add_argument("--router-entropy-max", type=float, default=0.0)
    parser.add_argument("--max-reload-drift", type=float, default=1.0e-5)
    parser.add_argument("--allow-nan-inf", action="store_true", help="Do not fail when NaN/Inf counters are non-zero.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = {
        "require_zero_nan_inf": not args.allow_nan_inf,
        "min_layer_coverage": args.min_layer_coverage,
        "router_entropy_min": args.router_entropy_min,
        "router_entropy_max": args.router_entropy_max,
        "max_reload_drift": args.max_reload_drift,
    }
    report = build_report(
        args.audit,
        metrics_paths=args.metrics,
        run_summary_paths=args.run_summary,
        thresholds=thresholds,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)
    return 0 if report["verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
