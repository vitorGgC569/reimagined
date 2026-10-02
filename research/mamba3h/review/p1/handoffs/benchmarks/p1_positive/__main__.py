"""Run only new P1 artifacts; never refresh frozen parent/P0 manifests."""

import argparse
import json
import time
import unittest

from .artifacts import audit, prepare, preserve_receipt, sha_manifest, verify_sha


def test():
    from . import tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"ok": result.wasSuccessful(), "tests_run": result.testsRun, "wall_seconds": time.perf_counter() - started,
        "max_threads": 2, "native_or_gpu_used": False, "learned_jobs_executed": 0,
        "failures": [{"test": str(t), "traceback": details} for t, details in result.failures],
        "errors": [{"test": str(t), "traceback": details} for t, details in result.errors],
        "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped]}
    preserve_receipt("tests", report)
    sha_manifest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "prepare", "audit", "sha", "verify-sha"))
    args = parser.parse_args()
    try:
        if args.command == "test":
            report = test()
        elif args.command == "prepare":
            report = prepare()
        elif args.command == "audit":
            report = audit()
            report = {key: value for key, value in report.items() if key != "rows"}
        elif args.command == "sha":
            report = sha_manifest()
        else:
            report = verify_sha()
    except Exception as exc:
        preserve_receipt("failure", {"status": "failed", "command": args.command,
                                     "exception": type(exc).__name__, "reason": str(exc)})
        raise
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report.get("ok", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
