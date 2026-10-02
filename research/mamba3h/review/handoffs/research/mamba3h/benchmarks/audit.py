"""Offline corpus audit, including actual rather than nominal overwrite counts."""

from collections import Counter
import hashlib
import json

from .controls import predict_reversed_order
from .manifest import ROOT, sha_manifest
from .oracle import solve
from .schema import IGNORE, UNDEFINED, fingerprint, model_view


def audit_episode(ep):
    view = model_view(ep)
    for field in ("tokens", "control", "query_mask"):
        if field in ep and ep[field] != view[field]:
            raise ValueError(f"Stored {field} differs from causal projection")
    if ep.get("targets") != solve(ep):
        raise ValueError("Stored targets differ from independent solver")
    alive, initialized, depth = {}, set(), {}
    result = Counter()
    for ev, label in zip(view["events"], ep["targets"]):
        ent, kind = ev["entity"], ev["kind"]
        result[kind] += 1
        if kind in ("SET", "WRITE"):
            if alive.get(ent):
                result["live_overwrite"] += 1
            elif ent in initialized:
                result["reinitialization_after_revoke"] += 1
            else:
                result["initialization"] += 1
            alive[ent], depth[ent] = True, 0
            initialized.add(ent)
        elif kind == "REVOKE":
            result["live_revocation" if alive.get(ent) else "already_absent_revocation"] += 1
            alive[ent], depth[ent] = False, 0
        elif kind == "OP":
            depth[ent] = depth.get(ent, 0) + 1
        elif kind == "QUERY":
            result["undefined_query" if label == UNDEFINED else "defined_query"] += 1
            if depth.get(ent, 0) >= 2 and label != UNDEFINED:
                result["defined_query_after_two_or_more_ops"] += 1
    if ep["task"] != "mqar":
        backwards = predict_reversed_order(ep)
        result["order_sensitive_query"] = sum(a != b for a, b in zip(backwards, ep["targets"]) if b != IGNORE)
    return dict(sorted(result.items()))


def audit_corpus():
    manifests = [("base", ROOT / "manifests" / "p0.json"), ("pilot-v1", ROOT / "manifests" / "pilot-v1-corpus.json")]
    seen_inputs, seen_seed_namespaces, rows, failures = {}, set(), [], []
    examples = 0
    for profile, path in manifests:
        if not path.exists():
            continue
        plan = json.loads(path.read_text("utf8"))
        for entry in plan["datasets"]:
            rel, counts, actual_inputs = entry["path"], Counter(), []
            data_path = (ROOT / rel).resolve()
            if not data_path.is_relative_to(ROOT):
                raise ValueError("Corpus manifest escapes owned root")
            raw = data_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                failures.append({"path": rel, "reason": "dataset_sha_mismatch"})
            episodes = [json.loads(line) for line in raw.splitlines() if line]
            if len(episodes) != entry["count"]:
                failures.append({"path": rel, "reason": "example_count_mismatch"})
            for ep in episodes:
                try:
                    counts.update(audit_episode(ep))
                except (ValueError, KeyError) as exc:
                    failures.append({"path": rel, "sample": ep.get("metadata", {}).get("sample"), "reason": str(exc)})
                inp = model_view(ep)
                fp = fingerprint(inp)
                if fp in seen_inputs:
                    failures.append({"path": rel, "reason": "duplicate_model_input", "other": seen_inputs[fp]})
                seen_inputs[fp] = rel
                meta = ep["metadata"]
                # Profile excluded intentionally: same causal RNG stream is allowed
                # in matched difficulty variants, split/seed sharing is not.
                seed_key = (profile, ep["task"], meta["derived_seed"])
                if seed_key in seen_seed_namespaces:
                    failures.append({"path": rel, "reason": "duplicate_derived_seed_namespace"})
                seen_seed_namespaces.add(seed_key)
                if (meta["split"], meta["seed"]) != (entry["split"], entry["seed"]):
                    failures.append({"path": rel, "reason": "split_seed_metadata_mismatch"})
                actual_inputs.append(inp)
                examples += 1
            if fingerprint(actual_inputs) != entry["input_fingerprint"]:
                failures.append({"path": rel, "reason": "input_fingerprint_mismatch"})
            rows.append({"profile": profile, "task": entry["task"], "seed": entry["seed"], "split": entry["split"],
                         "examples": len(episodes), "counts": dict(sorted(counts.items()))})
    result = {"kind": "executed_offline_corpus_audit", "ok": not failures, "failures": failures,
              "datasets": len(rows), "examples": examples, "unique_model_inputs": len(seen_inputs), "rows": rows,
              "overwrite_axis_semantics": "additional WRITE/SET count; live overwrite and post-revoke reinitialization reported separately"}
    (ROOT / "receipts").mkdir(exist_ok=True)
    (ROOT / "receipts" / "audit.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", "utf8")
    sha_manifest()
    return result
