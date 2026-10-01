"""Measurement comparisons must fail closed on artifacts, precision and policy."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from measure_decode_repro import POLICY_KEYS, verify_baseline_contract

class MeasurementContractTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "schema": "nsos-decode-measurement-v2", "status": "measured_not_quality_validated",
            "model": {"sha256": "weights"}, "tokenizer": {"sha256": "tokens"},
            "prompts": {"sha256": "prompts"}, "config": {"sha256": "config"},
            "options": {"temperature": 0}, "prompt_token_ids": [[1, 2]], "device": "gpu",
            "batch_size": 1, "packed_requested": False, "repeats": 5, "warmups": 2,
            "backend": "hip", "devices": [{"architecture": "gfx1102"}], "selected_device": 0,
            "matmul_precision_mode": 0, "parameter_count": 123,
            "policy_requests": {k: None for k in POLICY_KEYS},
        }

    def test_identical_contract(self):
        verify_baseline_contract(self.report, copy.deepcopy(self.report))

    def test_artifact_and_precision_changes_rejected(self):
        for key, value in (("model", {"sha256": "other"}), ("matmul_precision_mode", 1),
                           ("devices", [{"architecture": "gfx1100"}]), ("selected_device", 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.report)
                changed[key] = value
                with self.assertRaises(ValueError):
                    verify_baseline_contract(self.report, changed)

    def test_one_named_policy_only(self):
        changed = copy.deepcopy(self.report)
        changed["policy_requests"]["NSOS_GPU_GRAPH_DECODE"] = "1"
        with self.assertRaises(ValueError):
            verify_baseline_contract(self.report, changed)
        verify_baseline_contract(self.report, changed, "NSOS_GPU_GRAPH_DECODE")
        changed["policy_requests"]["NSOS_GPU_KV_DTYPE"] = "fp16"
        with self.assertRaises(ValueError):
            verify_baseline_contract(self.report, changed, "NSOS_GPU_GRAPH_DECODE")

    def test_incomplete_or_old_baseline_rejected(self):
        for key, value in (("status", "running"), ("schema", "nsos-decode-measurement-v1")):
            changed = copy.deepcopy(self.report)
            changed[key] = value
            with self.assertRaises(ValueError):
                verify_baseline_contract(changed, self.report)

if __name__ == "__main__":
    unittest.main()
