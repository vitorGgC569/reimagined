"""Reproducible corpus and bounded prepared factorial plan. No training here."""

from pathlib import Path
import hashlib
import json

from .generation import SEEDS, generate, pair_partition
from .oracle import solve
from .protocol import ARMS, BASELINE, LIMITS, MEASUREMENTS
from .schema import VERSION, canonical_bytes, fingerprint, model_view

ROOT = Path(__file__).resolve().parent
SPLIT_COUNTS = {"train": 64, "validation": 32, "test": 64}


def profiles():
    """Each within-group axis changes exactly one field from task's base."""
    result = []
    for task in ("mqar", "group", "inst"):
        base = dict(length=24, entities=1 if task == "group" else 4, operations=4,
                    distractors=4, overwrite=2, revocations=1, queries=3,
                    group="none" if task == "mqar" else "s5", regime="within_group")
        result.append({"task": task, "profile": "base", "difficulty": base, "changed_axis": None,
                       "status": "prepared", "evaluation": "within_group"})
        axes = dict(length=40, operations=8, distractors=8, overwrite=4, revocations=2)
        if task != "group":
            axes["entities"] = 8
        for axis, value in axes.items():
            result.append({"task": task, "profile": f"extrapolate_{axis}",
                           "difficulty": dict(base, **{axis: value}), "changed_axis": axis,
                           "status": "unattempted", "evaluation": "within_group_one_axis"})
        if task != "mqar":
            result.append({"task": task, "profile": "abelian_within_group",
                           "difficulty": dict(base, group="abelian"), "changed_axis": "group",
                           "status": "unattempted", "evaluation": "matched_commuting_task"})
            for regime in ("ordered_pairs", "cross_group"):
                result.append({"task": task, "profile": regime,
                               "difficulty": dict(base, regime=regime), "changed_axis": "regime",
                               "status": "unattempted", "evaluation": regime})
    return result


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def prepare():
    """Write owned paths only; idempotent and stable for frozen generator code."""
    entries, input_hashes, jobs = [], {}, []
    for task in ("mqar", "group", "inst"):
        for seed in SEEDS:
            for split, count in SPLIT_COUNTS.items():
                episodes = generate(task, seed, split, count)
                for ep in episodes:
                    if solve(ep) != ep["targets"]:
                        raise AssertionError("Independent solver disagreement; corpus not published")
                    key = fingerprint(model_view(ep))
                    if key in input_hashes:
                        raise AssertionError(f"Repeated model input across corpus: {input_hashes[key]}")
                    input_hashes[key] = (task, seed, split, ep["metadata"]["sample"])
                rel = f"datasets/{task}/seed-{seed}/{split}.jsonl"
                path = ROOT / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b"".join(canonical_bytes(ep) + b"\n" for ep in episodes)
                path.write_bytes(raw)
                entries.append({"task": task, "seed": seed, "split": split, "count": count,
                                "path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                "input_fingerprint": fingerprint([model_view(ep) for ep in episodes])})
            for arm in ARMS:
                datasets = {e["split"]: e["path"] for e in entries if e["task"] == task and e["seed"] == seed}
                jobs.append({"job_id": f"{task}:base:{arm}:{seed}", "task": task, "profile": "base",
                             "arm": arm, "seed": seed, "datasets": datasets,
                             "scheduled_initial_pilot": False, "status": "unattempted",
                             "accuracy": None, "reason": "prepared_only_root2_owns_training",
                             **{metric: None for metric in MEASUREMENTS},
                             "cpu_threads": 2, "hyperparameter_settings": 1})
    # Root2's narrower pilot is separately pinned, never silently substituted for
    # the prepared base grids above. Exact immutable bytes per seed/arm.
    pilot_path = ROOT.parent / "manifests" / "pilot-v1.json"
    pilot_datasets, pilot_jobs = [], []
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text("utf8"))
        if pilot["seeds"] != list(SEEDS) or pilot["arms"] != list(ARMS):
            raise ValueError("Root2 pilot seed/arm contract mismatch")
        for seed in SEEDS:
            for split, count in pilot["samples"].items():
                episodes = generate(pilot["task"], seed, split, count, **pilot["difficulty"])
                for ep in episodes:
                    if solve(ep) != ep["targets"]:
                        raise AssertionError("Pilot labels disagree with independent solver")
                    key = fingerprint(model_view(ep))
                    if key in input_hashes:
                        raise AssertionError("Pilot duplicate model input")
                    input_hashes[key] = ("pilot-v1", seed, split, ep["metadata"]["sample"])
                rel = f"datasets/pilot-v1/{pilot['task']}/seed-{seed}/{split}.jsonl"
                path = ROOT / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b"".join(canonical_bytes(ep) + b"\n" for ep in episodes)
                path.write_bytes(raw)
                pilot_datasets.append({"task": pilot["task"], "seed": seed, "split": split, "count": count,
                        "path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "input_fingerprint": fingerprint([model_view(ep) for ep in episodes])})
            for arm in ARMS:
                pilot_jobs.append({"job_id": f"{pilot['task']}:pilot-v1:{arm}:{seed}", "task": pilot["task"],
                    "profile": "pilot-v1", "arm": arm, "seed": seed, "status": "unattempted", "accuracy": None,
                    "reason": "prepared_only_root2_owns_training", "scheduled_initial_pilot": True,
                    "datasets": {e["split"]: e["path"] for e in pilot_datasets if e["seed"] == seed},
                    **{metric: None for metric in MEASUREMENTS}, "cpu_threads": 2, "hyperparameter_settings": 1})
        _write_json(ROOT / "manifests" / "pilot-v1-corpus.json", {
            "schema": VERSION, "root_protocol_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
            "root_protocol_path": "../manifests/pilot-v1.json", "protocol_snapshot": pilot,
            "datasets": pilot_datasets, "jobs": pilot_jobs, "label_mapping": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "125": 5}})
    contract = ROOT.parent / "integration" / "CONTRACT.md"
    contract_hash = hashlib.sha256(contract.read_bytes()).hexdigest() if contract.exists() else None
    plan = {"schema": VERSION, "baseline_commit": BASELINE, "contract_sha256_at_prepare": contract_hash,
            "baseline_kind_required": "actual_native_frozen_backbone_adapter_P0",
            "scope": "synthetic_component_correctness_and_prepared_factorial_only",
            "seeds": list(SEEDS), "arms": list(ARMS), "limits": LIMITS, "split_counts": SPLIT_COUNTS,
            "stopping_rules": ["stop_and_preserve_on_nonfinite", "censor_at_90_seconds_per_job",
                               "stop_aggregate_learned_budget_at_900_seconds", "never_replace_failed_seeds",
                               "test_never_selects_hyperparameters", "no_gpu_no_production_edits"],
            "gap_recovery_minimum_gap": 0.01, "paired_ci": {"method": "t", "df": 4, "confidence": 0.95,
                  "incomplete_policy": "undefined_CI_preserve_all_seeds_report_worst_case_bounds"},
            "arm_definitions": {"M0": "frozen_native_Mamba3", "M1": "M0_plus_noncommutative_dynamics",
                                "M2": "M0_plus_explicit_memory", "M3": "M0_plus_both",
                                "MA": "M0_plus_sparse_retrieval_attention", "MC": "matched_commuting_extra_capacity_control"},
            "matching": "same dataset bytes per arm/seed; record parameter/state/cache/reserved bytes and actual work residuals",
            "profiles": profiles(), "datasets": entries, "jobs": jobs,
            "ordered_pair_partition": {s: [list(p) for p in sorted(pair_partition(s))] for s in SPLIT_COUNTS},
            "ordered_pair_semantics": "per-entity chronological consecutive OPs, reset by SET or REVOKE",
            "cross_group_semantics": "train/validation S5, test Z5^3; separate exploratory evaluation, not within-group evidence",
            "evidence_limits": "No learned feasibility, integrated advantage, language/multimodal or SOTA claim"}
    _write_json(ROOT / "manifests" / "p0.json", plan)
    _write_json(ROOT / "manifests" / "results-template.json", jobs + pilot_jobs)
    sha_manifest()
    return {"datasets": len(entries) + len(pilot_datasets), "examples": len(input_hashes), "jobs": len(jobs) + len(pilot_jobs),
            "scheduled_pilot_jobs": len(pilot_jobs),
            "learned_jobs_executed_here": 0, "manifest": str(ROOT / "manifests" / "p0.json")}


def sha_manifest():
    """Pin owned source, corpora, protocol, handoff and receipts; exclude self/cache."""
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "SHA256.json":
            continue
        entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_json(ROOT / "manifests" / "SHA256.json", {"algorithm": "SHA256", "files": entries,
                 "aggregate_fingerprint": fingerprint(entries)})
    return entries


def verify_sha():
    manifest = json.loads((ROOT / "manifests" / "SHA256.json").read_text("utf8"))
    failed = []
    if fingerprint(manifest["files"]) != manifest["aggregate_fingerprint"]:
        failed.append("aggregate_fingerprint")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        # Reject a mutated manifest that escapes ownership.
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            failed.append(item["path"])
        elif path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            failed.append(item["path"])
    return {"ok": not failed, "files_checked": len(manifest["files"]), "failed": failed}
