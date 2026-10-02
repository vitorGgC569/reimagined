"""Bounded paired protocol; incomplete seeds never disappear from inference."""

import math
import statistics

SEEDS = (11, 23, 37, 53, 71)
ARMS = ("M0", "M1", "M2", "M3", "MA", "MC")
STATUSES = ("complete", "failed", "censored", "unattempted")
BASELINE = "4db1250e7435318e1c2522b27649a5d5433f2f48"
LIMITS = dict(cpu_threads=2, per_job_seconds=90, total_learned_seconds=900,
              max_updates=60, hyperparameter_settings=1, dim=8, length=24)
MEASUREMENTS = ("parameter_bytes", "persistent_state_bytes", "cache_bytes", "reserved_workspace_bytes",
                "retrieval_operations", "latency_seconds", "training_steps", "wall_seconds")


def accuracy(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target lengths differ")
    from .schema import IGNORE
    queries = [(p, t) for p, t in zip(predictions, targets) if t != IGNORE]
    if not queries:
        return {"accuracy": None, "correct": 0, "queries": 0}
    correct = sum(p == t for p, t in queries)
    return {"accuracy": correct / len(queries), "correct": correct, "queries": len(queries)}


def gap_recovery(candidate, baseline, ceiling=1.0, threshold=0.01):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in (candidate, baseline, ceiling)):
        raise ValueError("Finite accuracies in [0,1] required")
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("Invalid declared gap threshold")
    denominator = ceiling - baseline
    if denominator <= threshold:
        return {"defined": False, "value": None, "reason": "ceiling_minus_baseline_too_small",
                "denominator": denominator, "threshold": threshold}
    return {"defined": True, "value": (candidate - baseline) / denominator,
            "denominator": denominator, "threshold": threshold}


def validate_result(cell):
    if cell.get("arm") not in ARMS or cell.get("seed") not in SEEDS or cell.get("status") not in STATUSES:
        raise ValueError("Unknown arm, paired seed, or result status")
    score = cell.get("accuracy")
    if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("Accuracy must be finite [0,1] or null")
    if cell["status"] == "complete" and score is None:
        raise ValueError("Complete cell requires accuracy")
    if cell["status"] in ("failed", "censored") and not cell.get("reason"):
        raise ValueError("Failed/censored cell requires reason")
    for metric in MEASUREMENTS:
        value = cell.get(metric)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid measurement: {metric}")
    return cell


def paired_summary(cells, candidate="M3", baseline="M0"):
    """One task/profile only. All five seeds required for paired t inference.

    Incomplete pairs retain explicit statuses and worst-case [-1,+1] accuracy
    difference bounds. Never substitute partial/censored accuracy for completion.
    The t interval is small-sample descriptive uncertainty, not causal proof.
    """
    if candidate not in ARMS or baseline not in ARMS or candidate == baseline:
        raise ValueError("Distinct registered arms required")
    table = {}
    scope = {(c.get("task"), c.get("profile")) for c in cells}
    if len(scope) > 1:
        raise ValueError("Cannot silently pool tasks/profiles")
    for cell in cells:
        validate_result(cell)
        key = (cell["arm"], cell["seed"])
        if key in table:
            raise ValueError("Duplicate arm/seed result")
        table[key] = cell
    rows, deltas, lower, upper = [], [], [], []
    for seed in SEEDS:
        a = table.get((candidate, seed), {"status": "unattempted", "accuracy": None})
        b = table.get((baseline, seed), {"status": "unattempted", "accuracy": None})
        complete = a["status"] == b["status"] == "complete"
        delta = a["accuracy"] - b["accuracy"] if complete else None
        amin = a["accuracy"] if a["status"] == "complete" else 0
        amax = a["accuracy"] if a["status"] == "complete" else 1
        bmin = b["accuracy"] if b["status"] == "complete" else 0
        bmax = b["accuracy"] if b["status"] == "complete" else 1
        bounds = [amin - bmax, amax - bmin]
        lower.append(bounds[0]); upper.append(bounds[1])
        rows.append({"seed": seed, "candidate": dict(a), "baseline": dict(b), "delta": delta,
                     "delta_bounds": bounds})
        if complete:
            deltas.append(delta)
    result = {"candidate": candidate, "baseline": baseline, "seeds": rows,
              "complete_pairs": len(deltas), "required_pairs": 5,
              "mean_delta": None, "ci95": None, "df": None,
              "all_seed_mean_bounds": [sum(lower) / 5, sum(upper) / 5],
              "inference_defined": len(deltas) == 5}
    if len(deltas) == 5:
        mean = statistics.mean(deltas)
        half = 2.7764451051977987 * statistics.stdev(deltas) / math.sqrt(5)
        result.update(mean_delta=mean, ci95=[mean - half, mean + half], df=4)
    else:
        result["reason"] = "incomplete_five_seed_pairs_no_success_subset_inference"
    return result


def check_budget(cells, limits=None):
    limits = dict(LIMITS, **(limits or {}))
    violations, unknown = [], []
    wall = 0
    for c in cells:
        validate_result(c)
        name = f"{c.get('task', 'unspecified')}:{c.get('profile', 'base')}:{c['arm']}:{c['seed']}"
        for key, maximum in (("wall_seconds", limits["per_job_seconds"]), ("training_steps", limits["max_updates"]),
                             ("cpu_threads", limits["cpu_threads"]), ("hyperparameter_settings", limits["hyperparameter_settings"])):
            v = c.get(key)
            if v is None:
                unknown.append(f"{name}:{key}")
            elif not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > maximum:
                violations.append(f"{name}:{key}={v}, limit={maximum}")
        if c.get("wall_seconds") is not None:
            wall += c["wall_seconds"]
        for metric in MEASUREMENTS:
            if c.get(metric) is None:
                unknown.append(f"{name}:{metric}")
    if wall > limits["total_learned_seconds"]:
        violations.append(f"aggregate wall_seconds={wall}, limit={limits['total_learned_seconds']}")
    return {"within_known_limits": not violations, "fully_measured": not unknown,
            "violations": violations, "unknown": sorted(set(unknown)), "wall_seconds": wall, "limits": limits}


def factorial_summary(cells):
    """Optional raw-accuracy interaction M3-M1-M2+M0, preserving missing seeds."""
    # Validate duplicate/scope rules through the ordinary paired reports first.
    reports = {arm: paired_summary(cells, arm) for arm in ("M1", "M2", "M3", "MA", "MC")}
    table = {(c["arm"], c["seed"]): c for c in cells}
    rows = []
    for seed in SEEDS:
        items = [table.get((arm, seed)) for arm in ("M0", "M1", "M2", "M3")]
        value = None
        if all(c is not None and c["status"] == "complete" for c in items):
            m0, m1, m2, m3 = [c["accuracy"] for c in items]
            value = m3 - m1 - m2 + m0
        rows.append({"seed": seed, "interaction": value,
                     "statuses": {arm: table.get((arm, seed), {"status": "unattempted"})["status"] for arm in ARMS}})
    values = [r["interaction"] for r in rows]
    complete = all(v is not None for v in values)
    mean, interval = None, None
    if complete:
        mean = statistics.mean(values)
        half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
        interval = [mean - half, mean + half]
    return {"paired": reports, "interaction_seeds": rows, "interaction_mean": mean,
            "interaction_ci95": interval, "interaction_defined": complete,
            "interpretation": "Optional descriptive evidence; not required for complementarity or proof of mechanism"}
