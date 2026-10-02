"""Run from repository root: python -m research.mamba3h.benchmarks ..."""

import argparse
import json
import time
import unittest

from .manifest import ROOT, prepare, sha_manifest, verify_sha
from .schema import canonical_bytes, UNDEFINED


def smoke():
    from .generation import generate, SEEDS
    from .oracle import solve
    from .controls import predict_undefined, predict_first_write, predict_reversed_order
    from .protocol import accuracy
    start = time.perf_counter()
    rows = []
    for task, group in (("mqar", "none"), ("group", "s5"), ("group", "abelian"), ("inst", "s5"), ("inst", "abelian")):
        for seed in SEEDS:
            episodes = generate(task, seed, "test", 16, group=group)
            scores, negative, stale, reverse, queries, undefined = [], [], [], [], 0, 0
            for ep in episodes:
                predicted = solve(ep)
                if predicted != ep["targets"]:
                    raise AssertionError("Online labels disagree with independent solver")
                scores.append(accuracy(predicted, ep["targets"])["accuracy"])
                negative.append(accuracy(predict_undefined(ep), ep["targets"])["accuracy"])
                if task == "mqar":
                    stale.append(accuracy(predict_first_write(ep), ep["targets"])["accuracy"])
                else:
                    reverse.append(accuracy(predict_reversed_order(ep), ep["targets"])["accuracy"])
                queries += sum(ep["query_mask"])
                undefined += sum(v == UNDEFINED for v in ep["targets"])
            rows.append({"task": task, "group": group, "seed": seed, "episodes": len(episodes),
                         "queries": queries, "undefined_queries_included": undefined,
                         "oracle_accuracy": sum(scores) / len(scores),
                         "undefined_control_accuracy": sum(negative) / len(negative),
                         "stale_control_accuracy": sum(stale) / len(stale) if stale else None,
                         "reversed_order_accuracy": sum(reverse) / len(reverse) if reverse else None})
    report = {"kind": "executed_component_smoke_not_learned_factorial", "cpu_threads_max": 2,
              "wall_seconds": time.perf_counter() - start, "rows": rows,
              "failed": [], "censored": [], "learned_jobs_executed": 0}
    (ROOT / "receipts").mkdir(exist_ok=True)
    (ROOT / "receipts" / "smoke.json").write_bytes(canonical_bytes(report) + b"\n")
    sha_manifest()
    return report


def run_tests():
    from . import tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"kind": "executed_unit_tests", "tests_run": result.testsRun,
              "wall_seconds": time.perf_counter() - start, "cpu_threads_max": 2,
              "failures": [{"test": str(t), "traceback": detail} for t, detail in result.failures],
              "errors": [{"test": str(t), "traceback": detail} for t, detail in result.errors],
              "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
              "ok": result.wasSuccessful()}
    (ROOT / "receipts").mkdir(exist_ok=True)
    run_index = len(list((ROOT / "receipts").glob("tests-run-*.json"))) + 1
    (ROOT / "receipts" / f"tests-run-{run_index:03d}.json").write_bytes(canonical_bytes(report) + b"\n")
    (ROOT / "receipts" / "tests.json").write_bytes(canonical_bytes(report) + b"\n")
    sha_manifest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "prepare", "smoke", "audit", "verify-sha", "sha", "report"))
    parser.add_argument("--results", help="JSON array of learned result cells; report only, never trains")
    parser.add_argument("--task", choices=("mqar", "group", "inst"))
    parser.add_argument("--profile", default="base")
    args = parser.parse_args()
    if args.command == "test":
        report = run_tests()
    elif args.command == "prepare":
        report = prepare()
    elif args.command == "smoke":
        report = smoke()
    elif args.command == "audit":
        from .audit import audit_corpus
        report = audit_corpus()
        # Full per-split audit remains in the receipt, keep terminal output small.
        report = {k: v for k, v in report.items() if k != "rows"}
    elif args.command == "verify-sha":
        report = verify_sha()
    elif args.command == "sha":
        report = {"files_pinned": len(sha_manifest())}
    else:
        from .protocol import factorial_summary, check_budget
        if not args.results or not args.task:
            parser.error("report requires --results and --task; task/profile pooling is forbidden")
        with open(args.results, encoding="utf8") as stream:
            all_cells = json.load(stream)
        cells = [c for c in all_cells if c.get("task") == args.task and c.get("profile") == args.profile]
        report = {"task": args.task, "profile": args.profile, "factorial": factorial_summary(cells),
                  "budget": check_budget(all_cells)}
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report.get("ok", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
