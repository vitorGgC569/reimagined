#!/usr/bin/env python3
"""Fail-closed NSOS GPU release validation.

This runner is intentionally stricter than the ordinary developer CTest flow.
It requires a real CUDA device, builds only CUDA release-gate targets, executes
the complete GPU suite, proves the DP4A/checkpoint/mixed-precision contracts,
runs Compute Sanitizer memcheck, and emits a hash-addressed evidence bundle.

The process exits non-zero on every missing prerequisite, unsupported device,
failed assertion, skipped test, sanitizer finding, or incomplete evidence file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Sequence


GPU_TEST_TARGETS = (
    "test_gpu_parity",
    "test_mamba_parallel_scan_parity",
    "test_gpu_parity_basic",
    "test_gpu_parity_bitlinear",
    "test_gpu_parity_mamba_stream",
    "test_gpu_parity_mamba_scan",
    "test_gpu_parity_mamba_proper",
    "test_gpu_parity_mamba_proper_stream",
    "test_gpu_parity_mamba_nstate_stream",
    "test_gpu_parity_decode_incremental",
    "test_gpu_parity_mamba_nstate",
    "test_gpu_parity_mamba_faithful",
    "test_gpu_parity_jamba",
    "test_gpu_parity_moe_router",
    "test_gpu_parity_moe_batched",
    "test_gpu_parity_bitnet_dispatch",
    "test_gpu_parity_kan",
    "test_gpu_parity_sparse_attention",
    "test_gpu_parity_jamba_sparse",
    "test_gpu_parity_bitlinear_dp4a",
    "test_gpu_model_pack_dp4a_lifecycle",
    "test_gpu_device_memory_contract",
    "test_gpu_checkpoint_v8_continuation",
    "test_gpu_mixed_precision_contract",
)

GPU_BUILD_TARGETS = (*GPU_TEST_TARGETS, "bench_gpu_vs_cpu")

SANITIZER_TARGETS = (
    "test_gpu_device_memory_contract",
    "test_gpu_model_pack_dp4a_lifecycle",
    "test_gpu_parity_decode_incremental",
    "test_gpu_checkpoint_v8_continuation",
    "test_gpu_mixed_precision_contract",
)

STRICT_GPU_ENV = {
    "NSOS_CUDA_SYNC": "1",
    "NSOS_REQUIRE_GPU_TESTS": "1",
    "NSOS_GPU_MEMORY": "device",
    "NSOS_NO_MEMADVISE": "1",
}


class ValidationFailure(RuntimeError):
    """An expected fail-closed validation failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise ValidationFailure(f"required executable is unavailable: {name}")
    return resolved


def run_capture(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str],
    log_path: pathlib.Path | None = None,
    check: bool = True,
) -> tuple[int, str]:
    rendered = quote_command(command)
    print(f"+ {rendered}", flush=True)
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []
    log_stream = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8", newline="\n")
        log_stream.write(f"$ {rendered}\n")
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
            if log_stream is not None:
                log_stream.write(line)
        return_code = process.wait()
    finally:
        if log_stream is not None:
            log_stream.close()
    output = "".join(chunks)
    if check and return_code != 0:
        raise ValidationFailure(
            f"command returned {return_code}: {rendered}"
        )
    return return_code, output


def git_output(
    source_dir: pathlib.Path,
    args: Sequence[str],
    env: dict[str, str],
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=source_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def parse_compute_capability(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", value)
    if not match:
        raise ValidationFailure(
            f"nvidia-smi returned an invalid compute capability: {value!r}"
        )
    return int(match.group(1)), int(match.group(2))


def collect_preflight(
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    env: dict[str, str],
    *,
    required_gpu_name: str,
    minimum_compute_capability: tuple[int, int],
) -> dict[str, Any]:
    if sys.platform != "linux":
        raise ValidationFailure(
            f"gold GPU validation requires Linux, active platform is {sys.platform}"
        )

    for tool in ("git", "cmake", "ninja", "nvcc", "nvidia-smi"):
        require_tool(tool)
    sanitizer_path = require_tool("compute-sanitizer")

    _, gpu_csv = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=source_dir,
        env=env,
        log_path=output_dir / "preflight-nvidia-smi.log",
    )
    gpu_rows = [
        [field.strip() for field in line.split(",")]
        for line in gpu_csv.splitlines()
        if line.strip()
    ]
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 3:
        raise ValidationFailure(
            "gold validation requires exactly one visible CUDA GPU; "
            f"nvidia-smi rows={gpu_rows!r}"
        )
    gpu_name, compute_capability_text, driver_version = gpu_rows[0]
    compute_capability = parse_compute_capability(compute_capability_text)
    if compute_capability < minimum_compute_capability:
        raise ValidationFailure(
            "GPU compute capability is below the fail-closed minimum: "
            f"active={compute_capability_text}, "
            f"required={minimum_compute_capability[0]}."
            f"{minimum_compute_capability[1]}"
        )
    if required_gpu_name and required_gpu_name.casefold() not in gpu_name.casefold():
        raise ValidationFailure(
            f"required GPU name {required_gpu_name!r}, active GPU is {gpu_name!r}"
        )

    commit = git_output(source_dir, ["rev-parse", "HEAD"], env)
    branch = git_output(
        source_dir, ["rev-parse", "--abbrev-ref", "HEAD"], env
    )
    tracked_status = git_output(
        source_dir, ["status", "--porcelain", "--untracked-files=no"], env
    )
    if tracked_status:
        raise ValidationFailure(
            "tracked source tree is dirty; refusing non-reproducible validation:\n"
            + tracked_status
        )

    _, nvcc_version = run_capture(
        ["nvcc", "--version"],
        cwd=source_dir,
        env=env,
        log_path=output_dir / "preflight-nvcc.log",
    )
    _, sanitizer_version = run_capture(
        [sanitizer_path, "--version"],
        cwd=source_dir,
        env=env,
        log_path=output_dir / "preflight-compute-sanitizer.log",
    )
    _, cmake_version = run_capture(
        ["cmake", "--version"],
        cwd=source_dir,
        env=env,
        log_path=output_dir / "preflight-cmake.log",
    )

    return {
        "gpu": {
            "name": gpu_name,
            "compute_capability": compute_capability_text,
            "driver_version": driver_version,
            "count": 1,
        },
        "git": {
            "commit": commit,
            "branch": branch,
            "tracked_tree_clean": True,
        },
        "software": {
            "python": sys.version.splitlines()[0],
            "cmake": cmake_version.splitlines()[0],
            "nvcc": nvcc_version.strip(),
            "compute_sanitizer": sanitizer_version.strip(),
            "compute_sanitizer_path": sanitizer_path,
        },
    }


def configure_and_build(
    source_dir: pathlib.Path,
    build_dir: pathlib.Path,
    output_dir: pathlib.Path,
    env: dict[str, str],
    *,
    jobs: int,
    architecture: str,
) -> None:
    configure_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DNSOS_BUILD_TESTS=ON",
        "-DNSOS_BUILD_PYTHON=OFF",
        "-DNSOS_BUILD_API=OFF",
        "-DNSOS_BUILD_CLI=OFF",
        "-DNSOS_BUILD_OXTAMEM=OFF",
        "-DNSOS_ENABLE_CUDA=ON",
        "-DNSOS_ENABLE_NATIVE_OPTIMIZATIONS=OFF",
        f"-DNSOS_CUDA_ARCHITECTURES={architecture}",
    ]
    run_capture(
        configure_command,
        cwd=source_dir,
        env=env,
        log_path=output_dir / "cmake-configure.log",
    )
    run_capture(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(jobs),
            "--target",
            *GPU_BUILD_TARGETS,
        ],
        cwd=source_dir,
        env=env,
        log_path=output_dir / "cmake-build.log",
    )


def verify_junit(junit_path: pathlib.Path) -> dict[str, int]:
    text = junit_path.read_text(encoding="utf-8")

    def required_attribute(name: str) -> int:
        match = re.search(rf'\b{name}="(\d+)"', text)
        if not match:
            raise ValidationFailure(f"JUnit lacks required attribute {name!r}")
        return int(match.group(1))

    counts = {
        "tests": required_attribute("tests"),
        "failures": required_attribute("failures"),
        "disabled": required_attribute("disabled"),
        "skipped": required_attribute("skipped"),
    }
    expected = len(GPU_TEST_TARGETS)
    if counts != {
        "tests": expected,
        "failures": 0,
        "disabled": 0,
        "skipped": 0,
    }:
        raise ValidationFailure(f"GPU JUnit is not fail-closed green: {counts}")
    return counts


def run_gpu_suite(
    source_dir: pathlib.Path,
    build_dir: pathlib.Path,
    output_dir: pathlib.Path,
    env: dict[str, str],
) -> tuple[dict[str, int], dict[str, str]]:
    junit_path = output_dir / "gpu-ctest.xml"
    ctest_log = output_dir / "gpu-ctest.log"
    run_capture(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-R",
            r"^(test_gpu_|test_mamba_parallel_scan_parity$)",
            "--output-on-failure",
            "--output-junit",
            str(junit_path),
        ],
        cwd=source_dir,
        env=env,
        log_path=ctest_log,
    )
    counts = verify_junit(junit_path)

    contract_outputs: dict[str, str] = {}
    for target in (
        "test_gpu_model_pack_dp4a_lifecycle",
        "test_gpu_checkpoint_v8_continuation",
        "test_gpu_mixed_precision_contract",
    ):
        _, output = run_capture(
            [str(build_dir / target)],
            cwd=source_dir,
            env=env,
            log_path=output_dir / f"{target}.log",
        )
        contract_outputs[target] = output

    dp4a_output = contract_outputs["test_gpu_model_pack_dp4a_lifecycle"]
    if not re.search(r"\bload_dispatches=1\s+clone_dispatches=1\b", dp4a_output):
        raise ValidationFailure(
            "DP4A lifecycle evidence is missing load_dispatches=1 "
            "and clone_dispatches=1"
        )
    checkpoint_output = contract_outputs["test_gpu_checkpoint_v8_continuation"]
    if "version=8" not in checkpoint_output or "PASS" not in checkpoint_output:
        raise ValidationFailure("checkpoint v8 CUDA evidence is incomplete")
    mixed_output = contract_outputs["test_gpu_mixed_precision_contract"]
    if "fp16=executed" not in mixed_output:
        raise ValidationFailure(
            "FP16 was not actually executed; an unsupported-device rejection "
            "cannot satisfy the T4 mixed-precision gate"
        )
    if "fp16=rejected_as_unsupported" in mixed_output:
        raise ValidationFailure("FP16 was rejected instead of executed")
    if "PASS" not in mixed_output:
        raise ValidationFailure("mixed-precision CUDA contract did not pass")
    return counts, contract_outputs


def run_sanitizer(
    source_dir: pathlib.Path,
    build_dir: pathlib.Path,
    output_dir: pathlib.Path,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    sanitizer = require_tool("compute-sanitizer")
    results: dict[str, dict[str, Any]] = {}
    for target in SANITIZER_TARGETS:
        log_path = output_dir / f"compute-sanitizer-{target}.log"
        return_code, output = run_capture(
            [
                sanitizer,
                "--tool",
                "memcheck",
                "--leak-check",
                "full",
                "--error-exitcode",
                "99",
                "--print-session-details",
                str(build_dir / target),
            ],
            cwd=source_dir,
            env=env,
            log_path=log_path,
            check=False,
        )
        summary_matches = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", output)
        leak_matches = re.findall(
            r"LEAK SUMMARY:\s*(\d+)\s+bytes leaked", output
        )
        error_count = int(summary_matches[-1]) if summary_matches else None
        leak_bytes = int(leak_matches[-1]) if leak_matches else None
        results[target] = {
            "return_code": return_code,
            "error_count": error_count,
            "leaked_bytes": leak_bytes,
            "log": log_path.name,
        }
        if return_code != 0:
            raise ValidationFailure(
                f"Compute Sanitizer returned {return_code} for {target}"
            )
        if error_count != 0:
            raise ValidationFailure(
                f"Compute Sanitizer reported {error_count!r} errors for {target}"
            )
        if leak_bytes != 0:
            raise ValidationFailure(
                f"Compute Sanitizer reported {leak_bytes!r} leaked bytes "
                f"for {target}"
            )
    return results


def artifact_hashes(output_dir: pathlib.Path) -> list[dict[str, Any]]:
    excluded = {"result.json", "report.md", "SHA256SUMS"}
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in excluded:
            continue
        artifacts.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def write_evidence(
    output_dir: pathlib.Path,
    state: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state["artifacts"] = artifact_hashes(output_dir)
    json_path = output_dir / "result.json"
    json_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    gpu = state.get("preflight", {}).get("gpu", {})
    git = state.get("preflight", {}).get("git", {})
    junit = state.get("gpu_ctest", {})
    report_lines = [
        "# NSOS GPU Gold Validation",
        "",
        f"- Status: **{state.get('status', 'unknown').upper()}**",
        f"- Started: `{state.get('started_at', '')}`",
        f"- Finished: `{state.get('finished_at', '')}`",
        f"- Commit: `{git.get('commit', 'unavailable')}`",
        f"- Branch: `{git.get('branch', 'unavailable')}`",
        f"- GPU: `{gpu.get('name', 'unavailable')}`",
        f"- Compute capability: `{gpu.get('compute_capability', 'unavailable')}`",
        f"- Driver: `{gpu.get('driver_version', 'unavailable')}`",
        (
            "- GPU CTest: "
            f"`{junit.get('tests', 0)}` tests, "
            f"`{junit.get('failures', 0)}` failures, "
            f"`{junit.get('skipped', 0)}` skipped"
        ),
        f"- Compute Sanitizer targets: `{len(state.get('compute_sanitizer', {}))}`",
    ]
    if state.get("failure"):
        report_lines.extend(["", "## Failure", "", f"```text\n{state['failure']}\n```"])
    report_lines.extend(["", "## Artifact hashes", ""])
    for artifact in state["artifacts"]:
        report_lines.append(
            f"- `{artifact['sha256']}`  `{artifact['path']}` "
            f"({artifact['bytes']} bytes)"
        )
    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8", newline="\n"
    )

    final_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in final_paths),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    script_path = pathlib.Path(__file__).resolve()
    source_default = script_path.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=source_default,
        help="OXN/nsos source directory",
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=None,
        help="CUDA build directory (default: <source>/build-gpu-gold-sm75)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "evidence output directory "
            "(default: <source>/validation-artifacts/gpu-gold)"
        ),
    )
    parser.add_argument(
        "--require-gpu-name",
        default="",
        help="case-insensitive substring required in the active GPU name",
    )
    parser.add_argument(
        "--minimum-compute-capability",
        default="7.5",
        help="minimum CUDA compute capability (default: 7.5)",
    )
    parser.add_argument(
        "--cuda-architecture",
        default="75",
        help="CMake CUDA architecture (default: 75)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="parallel build jobs",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse an already configured and built directory",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = ()) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    build_dir = (
        args.build_dir.resolve()
        if args.build_dir is not None
        else source_dir / "build-gpu-gold-sm75"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else source_dir / "validation-artifacts" / "gpu-gold"
    )
    minimum_compute_capability = parse_compute_capability(
        args.minimum_compute_capability
    )
    if args.jobs <= 0:
        raise ValidationFailure("--jobs must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(STRICT_GPU_ENV)
    env["LC_ALL"] = "C.UTF-8"
    env["LANG"] = "C.UTF-8"

    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "requirements": {
            "required_gpu_name": args.require_gpu_name,
            "minimum_compute_capability": args.minimum_compute_capability,
            "cuda_architecture": args.cuda_architecture,
            "strict_gpu_environment": STRICT_GPU_ENV,
            "expected_gpu_tests": len(GPU_TEST_TARGETS),
            "expected_sanitizer_targets": list(SANITIZER_TARGETS),
        },
    }
    exit_code = 1
    try:
        state["preflight"] = collect_preflight(
            source_dir,
            output_dir,
            env,
            required_gpu_name=args.require_gpu_name,
            minimum_compute_capability=minimum_compute_capability,
        )
        if not args.skip_build:
            configure_and_build(
                source_dir,
                build_dir,
                output_dir,
                env,
                jobs=args.jobs,
                architecture=args.cuda_architecture,
            )
        state["gpu_ctest"], state["contract_outputs"] = run_gpu_suite(
            source_dir, build_dir, output_dir, env
        )
        state["compute_sanitizer"] = run_sanitizer(
            source_dir, build_dir, output_dir, env
        )
        state["status"] = "pass"
        exit_code = 0
    except Exception as error:  # evidence must survive every fail-closed exit
        state["status"] = "fail"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["traceback"] = traceback.format_exc()
        print(state["failure"], file=sys.stderr, flush=True)
    finally:
        state["finished_at"] = utc_now()
        write_evidence(output_dir, state)
        print(f"Evidence: {output_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
