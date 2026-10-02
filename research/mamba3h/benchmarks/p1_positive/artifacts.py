"""Owned P1 corpus/SHA publisher and read-only P0 freeze verification."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .generation import build_corpus, data_seed, input_fingerprint
from .oracle import solve
from .routing import retrieve
from .schema import COUNTS, DIFFICULTY, IGNORE, NAMESPACE, SEEDS, VERSION, canonical_bytes, fingerprint, model_numeric, model_view, validate_profile

ROOT = Path(__file__).resolve().parent
P0 = ROOT.parent
P0_AGGREGATE = "4802349ce71b88f6caa2bc9661b5b754aa753a4e83ba8879b16a506af4a29b7a"


def write_json(relative, value):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("P1 artifact write escapes ownership")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def preserve_receipt(name, value):
    directory = ROOT / "receipts"
    directory.mkdir(exist_ok=True)
    index = len(list(directory.glob(f"{name}-run-*.json"))) + 1
    write_json(f"receipts/{name}-run-{index:03d}.json", value)
    write_json(f"receipts/{name}.json", value)


def p0_frozen():
    """Verify all 79 old pinned files without refreshing any P0 manifest."""
    path = P0 / "manifests" / "SHA256.json"
    raw = path.read_bytes()
    old = json.loads(raw)
    failures = []
    if old["aggregate_fingerprint"] != P0_AGGREGATE or fingerprint(old["files"]) != P0_AGGREGATE:
        failures.append("P0 inventory fingerprint changed")
    for item in old["files"]:
        actual = (P0 / item["path"]).resolve()
        if not actual.is_relative_to(P0) or not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != item["sha256"]:
            failures.append(item["path"])
    return {"ok": not failures, "files_checked": len(old["files"]), "failures": failures,
            "aggregate_sha256": old["aggregate_fingerprint"], "manifest_file_sha256": hashlib.sha256(raw).hexdigest()}


def _p0_data_seeds():
    old = json.loads((P0 / "manifests" / "SHA256.json").read_text("utf8"))
    seeds = set()
    for item in old["files"]:
        if item["path"].startswith("datasets/") and item["path"].endswith(".jsonl"):
            for line in (P0 / item["path"]).read_bytes().splitlines():
                ep = json.loads(line)
                seeds.add(ep["metadata"]["derived_seed"])
    return seeds


def prepare():
    frozen_before = p0_frozen()
    if not frozen_before["ok"]:
        raise ValueError("P0 frozen receipt changed; not silently refreezing it")
    corpus, sampling = build_corpus()
    entries = []
    for (seed, split), examples in corpus.items():
        for ep in examples:
            if solve(ep) != ep["targets"]:
                raise ValueError("Independent solver mismatch; do not publish labels")
        relative = f"datasets/seed-{seed}/{split}.jsonl"
        raw = b"".join(canonical_bytes(ep) + b"\n" for ep in examples)
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        entries.append({"seed": seed, "split": split, "count": len(examples), "path": relative,
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "model_input_sha256": fingerprint([input_fingerprint(ep) for ep in examples]),
            "numeric_input_sha256": fingerprint([model_numeric(ep) for ep in examples])})
    manifest = {"schema": VERSION, "namespace": NAMESPACE, "status": "prepared_not_trained",
        "seeds": list(SEEDS), "counts": COUNTS, "difficulty": DIFFICULTY, "encoding_dim": 8,
        "encoding": "entity4 one-hot plus WRITE-only observed value4 one-hot; NOOP/QUERY value block is zero",
        "classes": [0, 1, 2, 3], "ignore": IGNORE, "query_mask": [False] * 5 + [True],
        "sampling": sampling, "datasets": entries, "p0_freeze": frozen_before,
        "freshness": "P1fresh namespaced sample/attempt seeds, bounded globally disjoint observable inputs across all splits/model seeds",
        "global_rejection_order": "fixed seed order, train/validation/test, sample index; <=16 attempts per sample",
        "resources": {"device": "cpu", "max_threads": 2, "max_examples": 800, "max_candidate_attempts": 12800,
                      "native_imports": False, "training_jobs_executed": 0},
        "protocol_owner": "Root2 freezes learned protocols, budgets, stopping and result cells before training",
        "evidence_scope": "fresh synthetic corpus and fixed exact-routing component correctness only"}
    write_json("manifests/corpus.json", manifest)
    result = audit(write=False)
    if not result["ok"]:
        preserve_receipt("prepare", {"status": "failed", "reason": "corpus_audit", "audit": result})
        raise ValueError("P1 corpus failed audit")
    after = p0_frozen()
    if after != frozen_before:
        raise ValueError("P0 changed during P1 publication")
    preserve_receipt("prepare", {"status": "complete", "datasets": len(entries), "examples": 800,
        "sampling": sampling, "audit_ok": True, "p0_unchanged": True, "learned_jobs_executed": 0})
    preserve_receipt("audit", result)
    sha_manifest()
    return {"ok": True, "datasets": len(entries), "examples": 800, "sampling": sampling,
            "p0_unchanged": True, "learned_jobs_executed": 0}


def audit(write=True):
    plan = json.loads((ROOT / "manifests" / "corpus.json").read_text("utf8"))
    seen_inputs, seen_seeds, failures, rows = set(), set(), [], []
    old_seeds = _p0_data_seeds()
    totals = Counter()
    for item in plan["datasets"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("P1 dataset escapes owned scope")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            failures.append({"path": item["path"], "reason": "file_sha"})
        episodes = [json.loads(line) for line in raw.splitlines() if line]
        if len(episodes) != item["count"] or item["count"] != COUNTS[item["split"]]:
            failures.append({"path": item["path"], "reason": "split_count"})
        inputs, numeric, counts = [], [], Counter()
        for sample, ep in enumerate(episodes):
            validate_profile(ep)
            if ep["targets"] != solve(ep) or ep["query_mask"] != [False] * 5 + [True]:
                failures.append({"path": item["path"], "sample": sample, "reason": "causal_targets_or_mask"})
            view = model_view(ep)
            for key in ("tokens", "control", "query_mask"):
                if ep[key] != view[key]:
                    failures.append({"path": item["path"], "sample": sample, "reason": f"stored_{key}"})
            fp = input_fingerprint(ep)
            if fp in seen_inputs:
                failures.append({"path": item["path"], "reason": "global_model_input_duplicate"})
            seen_inputs.add(fp)
            meta = ep["metadata"]
            if (meta["namespace"], meta["seed"], meta["split"], meta["sample"]) != (NAMESPACE, item["seed"], item["split"], sample):
                failures.append({"path": item["path"], "reason": "seed_split_namespace"})
            ds = meta["data_seed"]
            if ds != str(data_seed(item["seed"], item["split"], sample, meta["attempt"])) or not 0 <= meta["attempt"] < 16:
                failures.append({"path": item["path"], "reason": "derived_seed"})
            if ds in seen_seeds or ds in old_seeds:
                failures.append({"path": item["path"], "reason": "data_seed_duplicate_or_P0_reuse"})
            seen_seeds.add(ds)
            if meta["input_sha256"] != fp or meta["event_input_sha256"] != fingerprint(view):
                failures.append({"path": item["path"], "reason": "stored_input_fingerprint"})
            x = model_numeric(ep)
            if any(any(x[t][4:]) for t, e in enumerate(ep["events"]) if e["kind"] != "WRITE"):
                failures.append({"path": item["path"], "reason": "neutral_or_query_answer_leak"})
            outputs, state, work = retrieve(ep)
            decoded = max(range(4), key=lambda j: outputs[-1][4 + j])
            if outputs[-1][4:] != [float(j == ep["targets"][-1]) for j in range(4)] or decoded != ep["targets"][-1]:
                failures.append({"path": item["path"], "reason": "positive_routing_mismatch"})
            if sum(w["write_operations"] for w in work) != 3 or sum(w["read_operations"] for w in work) != 1 or sum(w["similarity_comparisons"] for w in work) != 3:
                failures.append({"path": item["path"], "reason": "routing_work_counters"})
            counts[f"class_{ep['targets'][-1]}"] += 1
            counts["examples"] += 1
            inputs.append(fp); numeric.append(x)
        if fingerprint(inputs) != item["model_input_sha256"] or fingerprint(numeric) != item["numeric_input_sha256"]:
            failures.append({"path": item["path"], "reason": "split_input_sha"})
        totals.update(counts)
        rows.append({"seed": item["seed"], "split": item["split"], "counts": dict(counts)})
    frozen = p0_frozen()
    if not frozen["ok"] or frozen != plan["p0_freeze"]:
        failures.append({"reason": "P0_freeze_changed"})
    result = {"ok": not failures, "kind": "executed_fresh_corpus_and_exact_routing_audit", "failures": failures,
        "datasets": len(rows), "examples": totals["examples"], "unique_model_inputs": len(seen_inputs),
        "unique_data_seeds": len(seen_seeds), "p0_data_seed_overlap": 0 if not seen_seeds & old_seeds else len(seen_seeds & old_seeds),
        "positive_routing_accuracy": 1.0 if not any(f.get("reason") == "positive_routing_mismatch" for f in failures) else None,
        "p0_unchanged": frozen["ok"], "class_counts": {k: v for k, v in totals.items() if k.startswith("class_")},
        "rows": rows, "learned_jobs_executed": 0, "native_or_gpu_used": False}
    if write:
        preserve_receipt("audit", result)
        sha_manifest()
    return result


def sha_manifest():
    files = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(ROOT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256.json"]
    write_json("manifests/SHA256.json", {"algorithm": "SHA256", "aggregate_sha256": fingerprint(files), "files": files})
    return {"files": len(files), "aggregate_sha256": fingerprint(files)}


def verify_sha():
    plan = json.loads((ROOT / "manifests" / "SHA256.json").read_text("utf8"))
    failures = []
    if fingerprint(plan["files"]) != plan["aggregate_sha256"]:
        failures.append("aggregate")
    paths = set()
    for item in plan["files"]:
        path = (ROOT / item["path"]).resolve()
        paths.add(item["path"])
        if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"] or path.stat().st_size != item["bytes"]:
            failures.append(item["path"])
    current = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256.json"}
    failures.extend(sorted(current ^ paths))
    frozen = p0_frozen()
    return {"ok": not failures and frozen["ok"], "files_checked": len(paths), "failures": failures,
            "aggregate_sha256": plan["aggregate_sha256"], "p0_unchanged": frozen["ok"]}
