#!/usr/bin/env python3
# scripts/gatekeeper.py - Validacao pre-deploy industrial

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GatekeeperCheck:
    name: str
    command: list[str]
    required: bool
    timeout: int = 300


class Gatekeeper:
    def __init__(self, min_coverage: float = 90.0):
        self.min_coverage = min_coverage
        self.results = []
        self.repo_root = Path(__file__).resolve().parents[2]
        self.oxn_root = self.repo_root / "OXN"
        self.nsos_root = self.oxn_root / "nsos"
        self.build_dir = self._resolve_build_dir()
        self.ctest_config = os.environ.get("NSOS_CTEST_CONFIG") or self._detect_ctest_config()
        self.report_path = self.oxn_root / "gatekeeper_report.json"
        self.env = os.environ.copy()
        python_path_entries = [
            str(self.build_dir),
            str(self.build_dir / "Release"),
            self.env.get("PYTHONPATH", ""),
        ]
        self.env["PYTHONPATH"] = os.pathsep.join([p for p in python_path_entries if p])
        self.checks = [
            GatekeeperCheck(
                "native_ctest",
                self._ctest_command(),
                True,
                timeout=900,
            ),
            GatekeeperCheck(
                "determinism",
                [
                    sys.executable,
                    str(self.oxn_root / "scripts" / "verify_determinism.py"),
                    "--build-dir",
                    str(self.build_dir),
                ],
                True,
            ),
            GatekeeperCheck(
                "industrial_python",
                [
                    sys.executable,
                    str(self.oxn_root / "tests" / "test_industrial_features.py"),
                ],
                True,
                timeout=600,
            ),
        ]

    def _resolve_build_dir(self) -> Path:
        explicit = os.environ.get("NSOS_BUILD_DIR")
        if explicit:
            return Path(explicit).expanduser().resolve()

        candidates = [
            self.nsos_root / "build-ci",
            self.nsos_root / "build-ci-local",
            self.nsos_root / "build-mvp",
            self.nsos_root / "build",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _detect_ctest_config(self) -> str | None:
        for config_name in ("Release", "RelWithDebInfo", "Debug"):
            if (self.build_dir / config_name).is_dir():
                return config_name
        return None

    def _ctest_command(self) -> list[str]:
        command = [
            "ctest",
            "--test-dir",
            str(self.build_dir),
            "--output-on-failure",
        ]
        if self.ctest_config:
            command.extend(["-C", self.ctest_config])
        return command

    def run_all(self) -> bool:
        print("=" * 60)
        print("NSOS GATEKEEPER - Pre-Deployment Validation")
        print("=" * 60)
        print(f"Repository root: {self.repo_root}")
        print(f"NSOS build dir: {self.build_dir}")
        if self.ctest_config:
            print(f"CTest config: {self.ctest_config}")

        all_passed = True
        for check in self.checks:
            print(f"Running: {check.name}...", flush=True)
            result = self._run_check(check)
            self.results.append(result)

            if not result["passed"] and check.required:
                all_passed = False
                print(f"REQUIRED CHECK FAILED: {check.name}")
            elif not result["passed"]:
                print(f"OPTIONAL CHECK FAILED: {check.name}")
            else:
                print(f"PASSED: {check.name}")

        self._generate_report()
        return all_passed

    def _run_check(self, check: GatekeeperCheck) -> dict:
        try:
            process = subprocess.run(
                check.command,
                capture_output=True,
                text=True,
                timeout=check.timeout,
                cwd=self.repo_root,
                env=self.env,
            )
            return {
                "name": check.name,
                "passed": process.returncode == 0,
                "command": check.command,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "required": check.required,
            }
        except subprocess.TimeoutExpired:
            return {
                "name": check.name,
                "passed": False,
                "command": check.command,
                "error": "Timeout",
                "required": check.required,
            }
        except Exception as exc:
            return {
                "name": check.name,
                "passed": False,
                "command": check.command,
                "error": str(exc),
                "required": check.required,
            }

    def _generate_report(self):
        with open(self.report_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)

        passed = sum(1 for result in self.results if result["passed"])
        total = len(self.results)
        print(f"\nSummary: {passed}/{total} checks passed")
        print(f"Detailed report saved to: {self.report_path}")


if __name__ == "__main__":
    gatekeeper = Gatekeeper(min_coverage=90.0)
    success = gatekeeper.run_all()
    sys.exit(0 if success else 1)
