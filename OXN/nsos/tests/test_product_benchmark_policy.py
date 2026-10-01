#!/usr/bin/env python3
"""Unit gates for Oxta Contábil benchmark resource policy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "scripts" / "oxta_contabil"
sys.path.insert(0, str(POLICY_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

import gpu_gold_validation as gpu_gold  # noqa: E402

from benchmark_policy import (  # noqa: E402
    configured_test_inventory,
    estimate_faithful_backward_atomics,
    resolve_gradient_checkpointing,
    validate_training_runtime_contract,
)
from benchmark_product_architecture import build_model  # noqa: E402
from benchmark_real_models_babi import (  # noqa: E402
    evenly_spaced_indices,
    parse_answer,
    response_schema,
)


class ProductBenchmarkPolicyTests(unittest.TestCase):
    def test_real_model_output_contract_is_closed_and_strict(self) -> None:
        vocabulary = {"hallway", "kitchen", "the"}
        schema = response_schema(vocabulary)
        self.assertEqual(
            schema["properties"]["answer"]["enum"],
            ["hallway", "kitchen", "the"],
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            parse_answer('{"answer":"Kitchen"}', vocabulary),
            "kitchen",
        )
        self.assertIsNone(
            parse_answer("The answer is kitchen.", vocabulary)
        )
        self.assertIsNone(
            parse_answer('{"answer":"bedroom"}', vocabulary)
        )

    def test_real_model_case_sampling_is_reproducible(self) -> None:
        self.assertEqual(
            evenly_spaced_indices(1000, 4), [0, 250, 500, 750]
        )
        self.assertEqual(
            evenly_spaced_indices(4, 4), [0, 1, 2, 3]
        )

    def test_product_model_fails_closed_on_determinism(self) -> None:
        class Config:
            pass

        class Model:
            def __init__(self, config, device) -> None:
                self.config = config
                self.device = device

            def to(self, device) -> None:
                self.device = device

        class FakeNsos:
            class HybridComposition:
                PARALLEL_GATED = "parallel"
                LEGACY_REPLACEMENT = "replacement"

            class Device:
                GPU = "gpu"

            ModelConfig = Config
            JambaModel = Model

            def __init__(self) -> None:
                self.seed = None
                self.deterministic = False

            def set_seed(self, seed: int) -> None:
                self.seed = seed

            def set_deterministic_reductions(self, enabled: bool) -> None:
                self.deterministic = enabled

            def deterministic_reductions_enabled(self) -> bool:
                return self.deterministic

        nsos = FakeNsos()
        _, config = build_model(
            nsos,
            "hybrid",
            vocab_size=64,
            seed=17,
            use_gradient_checkpointing=False,
        )
        self.assertEqual(nsos.seed, 17)
        self.assertTrue(nsos.deterministic)
        self.assertTrue(config.use_exact_attention_training)
        self.assertEqual(
            config.hybrid_composition,
            FakeNsos.HybridComposition.PARALLEL_GATED,
        )

    def test_configured_inventory_is_parsed_and_hashed(self) -> None:
        payload = (
            "format=nsos-ctest-inventory-v1\n"
            "source=/source\n"
            "gpu_backend=HIP\n"
            "python_extension=ON\n"
            "oxtamem_target=ON\n"
            "test_count=2\n"
            "tests_begin\n"
            "alpha\n"
            "beta\n"
            "tests_end\n"
            "quarantined_count=1\n"
            "quarantined_begin\n"
            "tests/legacy.cpp\n"
            "quarantined_end\n"
            "gpu_only_source_count=2\n"
            "gpu_only_sources_begin\n"
            "tests/gpu/alpha.cpp\n"
            "tests/gpu/beta.cpp\n"
            "gpu_only_sources_end\n"
            "oxtamem_only_source_count=1\n"
            "oxtamem_only_sources_begin\n"
            "tests/test_oxtamem_ffi.cpp\n"
            "oxtamem_only_sources_end\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nsos_test_inventory.txt"
            path.write_text(payload, encoding="utf-8")
            inventory = configured_test_inventory(Path(directory))
        self.assertEqual(inventory["tests"], ["alpha", "beta"])
        self.assertEqual(
            inventory["quarantined"], ["tests/legacy.cpp"]
        )
        self.assertEqual(
            inventory["gpu_only_sources"],
            ["tests/gpu/alpha.cpp", "tests/gpu/beta.cpp"],
        )
        self.assertEqual(
            inventory["oxtamem_only_sources"],
            ["tests/test_oxtamem_ffi.cpp"],
        )
        self.assertEqual(len(inventory["sha256"]), 64)

    def test_malformed_inventory_fails_closed(self) -> None:
        payloads = (
            "",
            (
                "format=nsos-ctest-inventory-v1\n"
                "test_count=1\n"
                "tests_begin\n"
                "tests_end\n"
                "quarantined_count=0\n"
                "quarantined_begin\n"
                "quarantined_end\n"
            ),
            (
                "format=nsos-ctest-inventory-v1\n"
                "test_count=2\n"
                "tests_begin\n"
                "beta\n"
                "alpha\n"
                "tests_end\n"
                "quarantined_count=0\n"
                "quarantined_begin\n"
                "quarantined_end\n"
            ),
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    path = (
                        Path(directory) / "nsos_test_inventory.txt"
                    )
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        configured_test_inventory(Path(directory))

    def test_rx7600_profile_retains_histories(self) -> None:
        policy = resolve_gradient_checkpointing(
            "auto",
            32,
            83,
            [
                {
                    "name": "AMD Radeon RX 7600",
                    "total_memory": 8_573_157_376,
                    "warp_size": 32,
                    "integrated": False,
                    "compiled": True,
                }
            ],
        )
        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["reason"], "auto_history_within_budget")
        self.assertEqual(policy["estimated_history_bytes"], 696_254_464)
        self.assertGreater(
            policy["history_budget_bytes"],
            policy["estimated_history_bytes"],
        )
        atomic = policy["faithful_backward_atomic_estimate"]
        self.assertEqual(atomic["scan_strategy"], "warp_aggregated")
        self.assertEqual(
            atomic["historical_total_atomics_all_layers"],
            388_333_568,
        )
        self.assertEqual(
            atomic["optimized_total_atomics_all_layers"],
            10_966_016,
        )
        self.assertGreater(atomic["estimated_reduction_ratio"], 35.0)

    def test_unaligned_head_uses_audited_fallback(self) -> None:
        atomic = estimate_faithful_backward_atomics(2, 7, 48)
        self.assertEqual(
            atomic["scan_strategy"], "scalar_atomic_fallback"
        )
        self.assertLess(
            atomic["optimized_total_atomics_all_layers"],
            atomic["historical_total_atomics_all_layers"],
        )

    def test_large_sequence_enables_checkpointing(self) -> None:
        policy = resolve_gradient_checkpointing(
            "auto",
            32,
            4096,
            [{
                "total_memory": 8_573_157_376,
                "integrated": False,
                "compiled": True,
            }],
        )
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["reason"], "auto_history_exceeds_budget")

    def test_unknown_or_integrated_only_vram_fails_safe(self) -> None:
        for devices in (
            [],
            [{"total_memory": 0, "integrated": False}],
            [{"total_memory": 16_000_000_000, "integrated": True}],
            [{
                "total_memory": 16_000_000_000,
                "integrated": False,
                "compiled": False,
            }],
        ):
            with self.subTest(devices=devices):
                policy = resolve_gradient_checkpointing(
                    "auto", 32, 83, devices
                )
                self.assertTrue(policy["enabled"])
                self.assertEqual(
                    policy["reason"], "auto_unknown_vram_fail_safe"
                )

    def test_explicit_modes_override_capacity(self) -> None:
        devices = [
            {
                "total_memory": 8_573_157_376,
                "integrated": False,
                "compiled": True,
            }
        ]
        self.assertTrue(
            resolve_gradient_checkpointing("on", 32, 83, devices)["enabled"]
        )
        self.assertFalse(
            resolve_gradient_checkpointing("off", 32, 4096, devices)[
                "enabled"
            ]
        )

    def test_invalid_contract_is_rejected(self) -> None:
        for requested, batch, sequence in (
            ("invalid", 32, 83),
            ("auto", 0, 83),
            ("auto", 32, 0),
        ):
            with self.subTest(
                requested=requested, batch=batch, sequence=sequence
            ):
                with self.assertRaises(ValueError):
                    resolve_gradient_checkpointing(
                        requested, batch, sequence, []
                    )

    def test_runtime_contract_bounds_d2h_to_control_scalars(self) -> None:
        contract = validate_training_runtime_contract(
            telemetry={"mamba_layers": []},
            transfers={
                "d2h_calls": 20,
                "d2h_bytes": 80,
                "device_synchronizations": 0,
                "stream_synchronizations": 20,
            },
            pool={
                "managed_cached_bytes": 0,
                "managed_live_bytes": 0,
                "managed_pressure_probes": 0,
                "cross_stream_domain_frees": 0,
            },
            arm="attention",
            steps=10,
            checkpoint_enabled=False,
        )
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["bounded_d2h_control_scalars"])
        self.assertTrue(
            contract["bounded_control_stream_synchronizations"]
        )

    def test_runtime_contract_bounds_deterministic_control_scalars(self) -> None:
        contract = validate_training_runtime_contract(
            telemetry={"mamba_layers": []},
            transfers={
                "d2h_calls": 40,
                "d2h_bytes": 200,
                "device_synchronizations": 0,
                "stream_synchronizations": 40,
            },
            pool={
                "managed_cached_bytes": 0,
                "managed_live_bytes": 0,
                "managed_pressure_probes": 0,
                "cross_stream_domain_frees": 0,
            },
            arm="attention",
            steps=10,
            checkpoint_enabled=False,
            deterministic_reductions=True,
        )
        self.assertTrue(contract["passed"])
        self.assertEqual(
            contract["allowed_d2h_control_scalars"]["max_calls"], 40
        )
        self.assertEqual(
            contract["allowed_d2h_control_scalars"]["max_bytes"], 200
        )
        self.assertEqual(
            contract["allowed_d2h_control_scalars"][
                "deterministic_grad_norm_f64_per_step"
            ],
            1,
        )

    def test_runtime_contract_rejects_bulk_d2h_in_deterministic_mode(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "bounded_d2h"):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 40,
                    "d2h_bytes": 201,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 40,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "cross_stream_domain_frees": 0,
                },
                arm="attention",
                steps=10,
                checkpoint_enabled=False,
                deterministic_reductions=True,
            )

    def test_runtime_contract_accepts_deterministic_mamba_reduction(
        self,
    ) -> None:
        contract = validate_training_runtime_contract(
            telemetry={
                "mamba_layers": [{}],
                "faithful_forward_gpu_calls": 2,
                "faithful_backward_gpu_calls": 2,
                "faithful_recompute_forwards": 0,
                "faithful_selective_history_recomputes": 0,
                "faithful_full_block_recompute_forwards": 0,
                "faithful_warp_aggregated_backward_calls": 0,
                "faithful_deterministic_backward_calls": 2,
                "faithful_scalar_atomic_backward_calls": 0,
                "faithful_reduced_conv_backward_calls": 2,
                "faithful_generic_atomic_conv_backward_calls": 0,
                "faithful_grouped_projection_forward_calls": 2,
                "faithful_grouped_projection_backward_calls": 2,
                "mamba_fast_path_fallbacks": 0,
                "faithful_forward_host_fallbacks": 0,
                "faithful_backward_host_fallbacks": 0,
            },
            transfers={
                "d2h_calls": 8,
                "d2h_bytes": 40,
                "device_synchronizations": 0,
                "stream_synchronizations": 8,
            },
            pool={
                "managed_cached_bytes": 0,
                "managed_live_bytes": 0,
                "managed_pressure_probes": 0,
                "cross_stream_domain_frees": 0,
            },
            arm="mamba",
            steps=2,
            checkpoint_enabled=False,
            deterministic_reductions=True,
        )
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["audited_scan_reduction_only"])

    def test_runtime_contract_rejects_per_sample_d2h(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bounded_d2h"):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 320,
                    "d2h_bytes": 1_280,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 0,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "cross_stream_domain_frees": 0,
                },
                arm="attention",
                steps=10,
                checkpoint_enabled=False,
            )

    def test_runtime_contract_allows_one_aggregate_qat_scalar(self) -> None:
        contract = validate_training_runtime_contract(
            telemetry={"mamba_layers": []},
            transfers={
                "d2h_calls": 30,
                "d2h_bytes": 120,
                "device_synchronizations": 0,
                "stream_synchronizations": 30,
            },
            pool={
                "managed_cached_bytes": 0,
                "managed_live_bytes": 0,
                "managed_pressure_probes": 0,
                "cross_stream_domain_frees": 0,
            },
            arm="hybrid_qat",
            steps=10,
            checkpoint_enabled=False,
        )
        self.assertTrue(contract["passed"])
        self.assertEqual(
            contract["allowed_d2h_control_scalars"][
                "qat_regularization_f32_per_step"
            ],
            1,
        )

    def test_runtime_contract_rejects_per_layer_qat_readbacks(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bounded_d2h"):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 31,
                    "d2h_bytes": 124,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 30,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "cross_stream_domain_frees": 0,
                },
                arm="hybrid_qat",
                steps=10,
                checkpoint_enabled=False,
            )

    def test_runtime_contract_rejects_excess_control_fences(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "bounded_control_stream_synchronizations"
        ):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 20,
                    "d2h_bytes": 80,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 21,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "cross_stream_domain_frees": 0,
                },
                arm="attention",
                steps=10,
                checkpoint_enabled=False,
            )

    def test_runtime_contract_rejects_pool_release_corruption(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "pool_integrity"):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 2,
                    "d2h_bytes": 8,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 0,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "cross_stream_domain_frees": 0,
                    "retained_release_bytes": 65536,
                    "retained_release_blocks": 1,
                    "release_failures": 1,
                    "unknown_deallocation_attempts": 0,
                    "capture_contract_violations": 0,
                },
                arm="attention",
                steps=1,
                checkpoint_enabled=False,
            )

    def test_runtime_contract_rejects_managed_advice_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "device_memory_only"):
            validate_training_runtime_contract(
                telemetry={"mamba_layers": []},
                transfers={
                    "d2h_calls": 2,
                    "d2h_bytes": 8,
                    "device_synchronizations": 0,
                    "stream_synchronizations": 0,
                },
                pool={
                    "managed_cached_bytes": 0,
                    "managed_live_bytes": 0,
                    "managed_pressure_probes": 0,
                    "managed_advice_failures": 1,
                    "cross_stream_domain_frees": 0,
                },
                arm="attention",
                steps=1,
                checkpoint_enabled=False,
            )


class GpuGoldInventoryTests(unittest.TestCase):
    """CPU-only release-gate regressions; no NSOS extension or GPU is loaded."""

    EXPECTED = ["test_gpu_attention_training_provider", "test_mamba_parallel_scan_parity"]

    @staticmethod
    def inventory(directory: Path, names: list[str]) -> None:
        entries = sorted(names)
        (directory / "nsos_test_inventory.txt").write_text(
            "format=nsos-ctest-inventory-v1\n"
            f"test_count={len(entries)}\n"
            "tests_begin\n" + "".join(name + "\n" for name in entries) +
            "tests_end\nquarantined_count=0\nquarantined_begin\nquarantined_end\n",
            encoding="utf-8",
        )

    @staticmethod
    def suite(names: list[str]) -> ET.Element:
        suite = ET.Element("testsuite", tests=str(len(names)), failures="0", disabled="0", skipped="0")
        for name in names:
            ET.SubElement(suite, "testcase", name=name, status="run", time="0.001")
        return suite

    def verify(self, suite: ET.Element, expected: list[str] | None = None) -> dict[str, int]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ctest.xml"
            ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
            return gpu_gold.verify_junit(path, self.EXPECTED if expected is None else expected)

    def test_gpu_gold_inventory_selects_expanded_graph_not_fixed_targets(self) -> None:
        names = [f"test_gpu_contract_{i:03d}" for i in range(74)]
        names.append("test_mamba_parallel_scan_parity")
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            self.inventory(build, names + ["test_cpu_policy", "test_mamba_parallel_scan_parity_extra", "prefix_test_gpu_fake"])
            self.assertEqual(gpu_gold.configured_gpu_test_names(build), sorted(names))
        self.assertEqual(self.verify(self.suite(names), names)["tests"], 75)

    def test_gpu_gold_inventory_missing_empty_and_duplicate_fail_closed(self) -> None:
        for names in (None, [], ["test_cpu_only"], [self.EXPECTED[0], self.EXPECTED[0]]):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                if names is not None:
                    self.inventory(build, names)
                with self.assertRaises(gpu_gold.ValidationFailure):
                    gpu_gold.configured_gpu_test_names(build)

    def test_gpu_gold_junit_accepts_actual_ctest_run_status(self) -> None:
        suite = self.suite(list(reversed(self.EXPECTED)))
        suite.set("errors", "0")
        self.assertEqual(self.verify(suite), {"tests": 2, "failures": 0, "disabled": 0, "skipped": 0})

    def test_gpu_gold_junit_rejects_missing_unknown_and_duplicate_names(self) -> None:
        cases = (
            self.EXPECTED[:1],
            [self.EXPECTED[0], "test_gpu_unknown"],
            [self.EXPECTED[0], self.EXPECTED[0]],
            self.EXPECTED + ["test_gpu_unknown"],
        )
        for names in cases:
            with self.subTest(names=names):
                suite = self.suite(names)
                suite.set("tests", "2")  # Forged green header must not hide case differences.
                with self.assertRaises(gpu_gold.ValidationFailure):
                    self.verify(suite)

    def test_gpu_gold_junit_rejects_duplicate_or_empty_expectations(self) -> None:
        for expected in ([], [self.EXPECTED[0], self.EXPECTED[0]]):
            with self.subTest(expected=expected), self.assertRaises(gpu_gold.ValidationFailure):
                self.verify(self.suite(expected), expected)

    def test_gpu_gold_junit_rejects_case_failures_errors_and_skips_with_green_header(self) -> None:
        for tag in ("failure", "error", "skipped"):
            for location in ("case", "suite"):
                with self.subTest(tag=tag, location=location):
                    suite = self.suite(self.EXPECTED)
                    parent = suite[0] if location == "case" else suite
                    ET.SubElement(parent, tag, message="hidden by adulterated header")
                    with self.assertRaises(gpu_gold.ValidationFailure):
                        self.verify(suite)

    def test_gpu_gold_junit_rejects_notrun_disabled_unknown_and_missing_status(self) -> None:
        for status in ("notrun", "disabled", "skipped", "unknown", "", None):
            with self.subTest(status=status):
                suite = self.suite(self.EXPECTED)
                if status is None:
                    del suite[0].attrib["status"]
                else:
                    suite[0].set("status", status)
                with self.assertRaises(gpu_gold.ValidationFailure):
                    self.verify(suite)

    def test_gpu_gold_junit_rejects_adulterated_counts(self) -> None:
        alterations = [("tests", value) for value in ("0", "1", "3", "NaN", "-2", "+2", "2.0")]
        alterations += [(key, "1") for key in ("failures", "disabled", "skipped", "errors")]
        alterations += [(key, "invalid") for key in ("failures", "disabled", "skipped", "errors")]
        for key, value in alterations:
            with self.subTest(key=key, value=value):
                suite = self.suite(self.EXPECTED)
                suite.set(key, value)
                with self.assertRaises(gpu_gold.ValidationFailure):
                    self.verify(suite)

    def test_gpu_gold_junit_requires_all_ctest_header_counts_including_disabled(self) -> None:
        for key in ("tests", "failures", "disabled", "skipped"):
            with self.subTest(key=key):
                suite = self.suite(self.EXPECTED)
                del suite.attrib[key]
                with self.assertRaises(gpu_gold.ValidationFailure):
                    self.verify(suite)

    def test_gpu_gold_junit_rejects_nested_cases_and_foreign_root(self) -> None:
        suite = self.suite(self.EXPECTED)
        case = suite[0]
        suite.remove(case)
        ET.SubElement(suite, "testsuite").append(case)
        with self.assertRaises(gpu_gold.ValidationFailure):
            self.verify(suite)
        suite = self.suite(self.EXPECTED)
        suite.tag = "testsuites"
        with self.assertRaises(gpu_gold.ValidationFailure):
            self.verify(suite)

    def test_gpu_gold_junit_rejects_missing_file_and_malformed_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ctest.xml"
            with self.assertRaises(gpu_gold.ValidationFailure):
                gpu_gold.verify_junit(path, self.EXPECTED)
            path.write_text("<testsuite", encoding="utf-8")
            with self.assertRaises(gpu_gold.ValidationFailure):
                gpu_gold.verify_junit(path, self.EXPECTED)

    @unittest.skipUnless(shutil.which("ctest"), "CTest executable unavailable for CPU-only regex check")
    def test_gpu_gold_regex_matches_actual_ctest_posix_filter(self) -> None:
        names = self.EXPECTED + ["test_cpu_policy", "test_mamba_parallel_scan_parity_extra", "prefix_test_gpu_fake"]
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CTestTestfile.cmake").write_text(
                "".join(f'add_test({name} "{Path(sys.executable).as_posix()}" "-c" "pass")\n' for name in names),
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["ctest", "--test-dir", str(build), "--show-only=json-v1", "-R", gpu_gold.GPU_CTEST_PATTERN],
                capture_output=True, text=True, check=True,
            )
            selected = [test["name"] for test in json.loads(completed.stdout)["tests"]]
        self.assertEqual(sorted(selected), sorted(self.EXPECTED))

    @unittest.skipUnless(shutil.which("ctest"), "CTest executable unavailable for CPU-only JUnit check")
    def test_gpu_gold_actual_ctest_pass_disabled_and_notrun_xml(self) -> None:
        # Generated Python/disabled/missing-command fixtures only, never GPU tests.
        name = self.EXPECTED[0]
        for fault in ("pass", "disabled", "notrun"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                build = Path(directory)
                command = Path(sys.executable) if fault != "notrun" else build / "missing-test-command"
                fixture = f'add_test({name} "{command.as_posix()}" "-c" "pass")\n'
                if fault == "disabled":
                    fixture += f'set_tests_properties({name} PROPERTIES DISABLED TRUE)\n'
                (build / "CTestTestfile.cmake").write_text(fixture, encoding="utf-8")
                path = build / "ctest.xml"
                subprocess.run(["ctest", "--test-dir", str(build), "--output-junit", str(path)],
                               capture_output=True, text=True, check=False)
                suite = ET.parse(path).getroot()
                case = suite.find("testcase")
                self.assertIsNotNone(case)
                if fault == "pass":
                    self.assertEqual(case.get("status"), "run")
                    self.assertEqual(gpu_gold.verify_junit(path, [name])["tests"], 1)
                    continue
                if fault == "disabled":
                    self.assertIn(case.get("status"), {"disabled", "notrun"})
                else:
                    self.assertEqual(case.get("status"), "notrun")
                with self.assertRaises(gpu_gold.ValidationFailure):
                    gpu_gold.verify_junit(path, [name])
                # Laundering headers and child flags cannot hide a non-run case.
                for key in ("failures", "disabled", "skipped", "errors"):
                    suite.set(key, "0")
                suite.set("tests", "1")
                for child in list(case):
                    if child.tag in {"failure", "error", "skipped"}:
                        case.remove(child)
                with self.assertRaises(gpu_gold.ValidationFailure):
                    self.verify(suite, [name])

    def test_gpu_gold_build_covers_configured_targets_without_static_target_subset(self) -> None:
        with mock.patch.object(gpu_gold, "run_capture") as capture:
            gpu_gold.configure_and_build(Path("source"), Path("build"), Path("output"), {},
                                         jobs=2, architecture="75", generator="Ninja")
        command = capture.call_args_list[-1].args[0]
        self.assertEqual(command[:2], ["cmake", "--build"])
        self.assertNotIn("--target", command)

    def test_gpu_gold_suite_uses_inventory_and_same_ctest_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            self.inventory(build, self.EXPECTED)
            ET.ElementTree(self.suite(self.EXPECTED)).write(build / "gpu-ctest.xml", encoding="utf-8")
            outputs = [
                (0, "CTest green"),
                (0, "load_dispatches=1 clone_dispatches=1"),
                (0, "version=10 PASS"),
                (0, "fp16=executed PASS"),
            ]
            with mock.patch.object(gpu_gold, "run_capture", side_effect=outputs) as capture:
                counts, _ = gpu_gold.run_gpu_suite(build, build, build, {})
            command = capture.call_args_list[0].args[0]
        self.assertEqual(counts["tests"], len(self.EXPECTED))
        self.assertEqual(command[command.index("-R") + 1], gpu_gold.GPU_CTEST_PATTERN)

    def test_gpu_gold_suite_rejects_missing_inventory_before_any_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(gpu_gold, "run_capture") as capture:
            build = Path(directory)
            with self.assertRaises(gpu_gold.ValidationFailure):
                gpu_gold.run_gpu_suite(build, build, build, {})
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
