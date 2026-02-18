#!/usr/bin/env python3
# scripts/gatekeeper.py - Validação pré-deploy industrial

import sys
import subprocess
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class GatekeeperCheck:
    name: str
    command: str
    required: bool
    timeout: int = 300

class Gatekeeper:
    CHECKS = [
        GatekeeperCheck("unit_tests", "python -m pytest tests", True),
        GatekeeperCheck("determinism", "python scripts/verify_determinism.py", True),
        # GatekeeperCheck("memory_leaks", "valgrind --leak-check=full ./build/test_suite", False),
    ]

    def __init__(self, min_coverage: float = 90.0):
        self.min_coverage = min_coverage
        self.results = []

    def run_all(self) -> bool:
        print("=" * 60)
        print("NSOS GATEKEEPER - Pre-Deployment Validation")
        print("=" * 60)

        all_passed = True
        for check in self.CHECKS:
            print(f"Running: {check.name}...", flush=True)
            result = self._run_check(check)
            self.results.append(result)

            if not result["passed"] and check.required:
                all_passed = False
                print(f"❌ REQUIRED CHECK FAILED: {check.name}")
            elif not result["passed"]:
                print(f"⚠️  OPTIONAL CHECK FAILED: {check.name}")
            else:
                print(f"✅ PASSED: {check.name}")

        self._generate_report()
        return all_passed

    def _run_check(self, check: GatekeeperCheck) -> dict:
        try:
            process = subprocess.run(
                check.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=check.timeout
            )
            return {
                "name": check.name,
                "passed": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "required": check.required
            }
        except subprocess.TimeoutExpired:
            return {
                "name": check.name,
                "passed": False,
                "error": "Timeout",
                "required": check.required
            }
        except Exception as e:
            return {
                "name": check.name,
                "passed": False,
                "error": str(e),
                "required": check.required
            }

    def _generate_report(self):
        report_path = "gatekeeper_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\nSummary: {passed}/{total} checks passed")
        print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    gatekeeper = Gatekeeper(min_coverage=90.0)
    success = gatekeeper.run_all()
    sys.exit(0 if success else 1)
