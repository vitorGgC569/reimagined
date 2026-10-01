from pathlib import Path
import sys
import unittest
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import bench_training_engineering as probe
from audit_training_integrations import missing_gradient_paths, validate_probe_geometry


class EngineeringProbeTests(unittest.TestCase):
    def test_sparse_activity_is_required_even_with_legacy_moe_compute(self):
        args = SimpleNamespace(gpu_training_profile="inherit", moe_compute_policy="legacy")
        with patch.dict(probe.os.environ, {}, clear=True):
            policy = probe.core.configure_gpu_training_profile(args, SimpleNamespace(use_moe=True), True)
            with self.assertRaisesRegex(RuntimeError, "sparse gradient policy mismatch"):
                probe.core.validate_native_training_policy(policy, {})
            self.assertFalse(policy["native_verified"])
            probe.core.validate_native_training_policy(policy, {
                "optimizer.sparse_gradient_policy": policy["moe_sparse_gradient"]})
            self.assertTrue(policy["native_verified"])

    def test_wmma_moe_requires_explicit_native_policy(self):
        args = SimpleNamespace(gpu_training_profile="inherit", moe_compute_policy="wmma-v1")
        config = SimpleNamespace(use_moe=True)
        with patch.dict(probe.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "wave32"):
                probe.core.configure_gpu_training_profile(args, config, True, 64)
            policy = probe.core.configure_gpu_training_profile(args, config, True, 32)
            identity = {"moe.dispatch_policy": "stable_expert_row_order_device_combine_v1",
                        "moe.training_compute_policy": policy["moe_compute"]}
            with self.assertRaisesRegex(RuntimeError, "NSOS_MOE_WMMA_TRAINING"):
                probe.core.validate_native_training_policy(policy, identity)
            self.assertFalse(policy["native_verified"])
            identity["moe.training_wmma_policy"] = policy["moe_wmma"]
            identity["optimizer.sparse_gradient_policy"] = policy["moe_sparse_gradient"]
            probe.core.validate_native_training_policy(policy, identity)
            self.assertTrue(policy["native_verified"])
            args.moe_compute_policy = "inherit"
            probe.os.environ["NSOS_MOE_WMMA_TRAINING"] = "1invalid"
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                probe.core.configure_gpu_training_profile(args, config, True, 32)
            args.moe_compute_policy = "grouped-v1"
            self.assertEqual(probe.core.configure_gpu_training_profile(args, config, True, 32)["moe_wmma"], "off")

    def test_grouped_moe_is_explicit_and_native_verified(self):
        args = SimpleNamespace(gpu_training_profile="inherit", moe_compute_policy="grouped-v1")
        config = SimpleNamespace(use_moe=False)
        with patch.dict(probe.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "use_moe"):
                probe.core.configure_gpu_training_profile(args, config, True)
            self.assertFalse(config.use_moe)
            config.use_moe = True
            with self.assertRaisesRegex(ValueError, "GPU"):
                probe.core.configure_gpu_training_profile(args, config, False)
            policy = probe.core.configure_gpu_training_profile(args, config, True)
            self.assertEqual(policy["switches"]["NSOS_MOE_ORDERED_DEVICE"], "1")
            self.assertFalse(policy["native_verified"])
            self.assertEqual(policy["moe_compute"], "device_segmented_tile16_active_qat_v2")
            with self.assertRaisesRegex(RuntimeError, "NSOS_MOE_GROUPED_TRAINING"):
                probe.core.validate_native_training_policy(policy, {
                    "moe.dispatch_policy": "stable_expert_row_order_device_combine_v1",
                    "moe.training_compute_policy": "device_segmented_tile16_late_registry_v1"})
            with self.assertRaisesRegex(RuntimeError, "NSOS_MOE_GROUPED_TRAINING"):
                probe.core.validate_native_training_policy(policy, {
                    "moe.dispatch_policy": "stable_expert_row_order_device_combine_v1"})
            probe.core.validate_native_training_policy(policy, {
                "moe.dispatch_policy": "stable_expert_row_order_device_combine_v1",
                "moe.training_compute_policy": policy["moe_compute"],
                "optimizer.sparse_gradient_policy": policy["moe_sparse_gradient"]})
            self.assertTrue(policy["native_verified"])
            args.moe_compute_policy = "inherit"
            probe.os.environ["NSOS_MOE_ORDERED_DEVICE"] = "0"
            with self.assertRaisesRegex(ValueError, "ordered routing"):
                probe.core.configure_gpu_training_profile(args, config, True)
            args.moe_compute_policy = "legacy"
            self.assertEqual(probe.core.configure_gpu_training_profile(args, config, True)["moe_compute"], "legacy")

    def test_native_policy_rejects_ignored_flags_and_stale_ttt_derivative(self):
        policy = {"switches": {"NSOS_TTT_FULL_BPTT": "1"},
                  "ttt_derivative": "full_sequence_bptt_isolated_boundary32_column_q_v2"}
        for identity in ({}, {"ttt.training_policy": "device_recurrence_fp64_norm_truncated_v1"},
                         {"ttt.training_policy": "full_sequence_bptt_isolated_boundary32_v1"}):
            with self.assertRaisesRegex(RuntimeError, "TTT training policy mismatch"):
                probe.core.validate_native_training_policy(policy, identity)
        probe.core.validate_native_training_policy(policy, {"ttt.training_policy": policy["ttt_derivative"]})
        policy = {"switches": {"NSOS_MAMBA_BOUNDARY_HISTORY": "1"}}
        with self.assertRaisesRegex(RuntimeError, "NSOS_MAMBA_BOUNDARY_HISTORY"):
            probe.core.validate_native_training_policy(policy, {})
        probe.core.validate_native_training_policy(policy, {"mamba.history_retention": "entering_boundary_chunk32_recompute_v1"})

    def test_integration_geometry_is_checked_before_loading_native_code(self):
        for width in (64, 128, 256, 768):
            validate_probe_geometry(512, 5, 2, width)
        for geometry in ((1, 5, 2, 128), (8193, 5, 2, 128), (512, 1, 2, 128),
                         (512, 5, 0, 128), (512, 5, 9, 128), (512, 5, 2, 0),
                         (512, 5, 2, 65), (512, 5, 2, 832)):
            with self.assertRaises(ValueError):
                validate_probe_geometry(*geometry)

    def test_full_ttt_policy_does_not_silently_change_architecture(self):
        args = SimpleNamespace(gpu_training_profile="inherit", ttt_gradient_policy="full-sequence-v1")
        with patch.dict(probe.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "use_ttt"):
                probe.core.configure_gpu_training_profile(args, SimpleNamespace(use_ttt=False), True)
            policy = probe.core.configure_gpu_training_profile(args, SimpleNamespace(use_ttt=True), True)
            self.assertEqual(policy["ttt_derivative"], "full_sequence_bptt_isolated_boundary32_column_q_v2")
            self.assertEqual(probe.os.environ["NSOS_TTT_FULL_BPTT"], "1")
            args.ttt_gradient_policy = "truncated"
            self.assertEqual(probe.core.configure_gpu_training_profile(args, SimpleNamespace(use_ttt=True), True)
                ["ttt_derivative"], "truncated_stop_gradient_adaptation")
    @unittest.skipUnless(os.name == "nt", "PowerShell launcher is Windows-only")
    def test_pilot_launcher_has_explicit_profile_and_unfenced_default(self):
        launcher = Path(probe.__file__).with_name("start_ptbr_verified_pilot.ps1")
        for profile, precision, policy in (("chunked-bf16", "bf16", "legacy"),
                                            ("redesign-bf16", "bf16", "redesign-v1"),
                                            ("redesign-fp32", "fp32", "redesign-v1")):
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
                str(launcher), "-Action", "train", "-Profile", profile, "-MaxTrainSteps", "20",
                "-RunName", "validation-dry-run", "-DryRun"], capture_output=True, text=True,
                timeout=20, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            plan = json.loads(result.stdout)
            self.assertEqual(plan["Precision"], precision)
            self.assertEqual(plan["GpuTrainingProfile"], policy)
            self.assertEqual(plan["TrainingTiming"], "0")
            self.assertEqual(plan["LayerTiming"], "0")
            self.assertEqual(plan["BackwardChunkSize"], "32")
            self.assertIn("--gpu-training-profile", plan["Arguments"])

    def test_default_measurement_has_no_diagnostic_fences(self):
        for name in probe.PROFILES:
            env = probe.profile_environment(name)
            self.assertEqual(env["NSOS_TRAIN_TIMING"], "0")
            self.assertEqual(env["NSOS_MAMBA_STAGE_TIMING"], "0")
            self.assertEqual(env["NSOS_DETERMINISTIC"], "1")

    def test_precision_comparison_keeps_identical_kernel_flags(self):
        self.assertEqual(probe.profile_environment("chunked-fp32"),
                         probe.profile_environment("chunked-bf16"))

    def test_invalid_profile_rejected(self):
        with self.assertRaises(ValueError):
            probe.profile_environment("unknown")

    def test_redesign_and_isolated_profiles_are_explicit(self):
        features = ("NSOS_MAMBA_BOUNDARY_HISTORY", "NSOS_DEVICE_GRAD_CLIP",
                    "NSOS_ATTN_TILED_TRAINING", "NSOS_MOE_ORDERED_DEVICE",
                    "NSOS_TTT_DEVICE_RECURRENCE")
        for feature in features:
            self.assertEqual(probe.profile_environment("chunked-bf16")[feature], "0")
            self.assertEqual(probe.profile_environment("redesign-bf16")[feature], "1")
        self.assertEqual(probe.profile_environment("boundary-bf16")[features[0]], "1")
        self.assertEqual(probe.profile_environment("boundary-bf16")[features[1]], "0")
        self.assertEqual(probe.profile_environment("clip-bf16")[features[0]], "0")
        self.assertEqual(probe.profile_environment("clip-bf16")[features[1]], "1")

    def test_delta_excludes_flags_and_preserves_zero(self):
        self.assertEqual(probe.numeric_delta({"a": 10, "b": 3, "flag": True},
                                            {"a": 4, "b": 3}), {"a": 6, "b": 0})

    def test_integration_requires_real_branch_gradients(self):
        paths = ["embedding.weight", "norm_f.weight", "layers.0.mamba.x_proj.weight",
                 "layers.1.attn.q_down_proj.weight", "layers.1.ffn.pre_norm.weight"]
        parameters = [{"name": name, "gradient_present": True, "gradient_nonzero": True} for name in paths]
        self.assertEqual(missing_gradient_paths("parallel", parameters), [])
        parameters[2]["gradient_nonzero"] = False
        self.assertIn(".mamba.", missing_gradient_paths("parallel", parameters))
        parameters[3]["gradient_present"] = False
        self.assertIn(".attn.", missing_gradient_paths("parallel", parameters))

    def test_weight_fingerprint_covers_layout_name_and_values(self):
        values = np.array([[1, 2]], dtype=np.float32)
        parameter = SimpleNamespace(name="a", data=SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: values)))
        model = SimpleNamespace(parameters=lambda: [parameter])
        original = probe.model_weight_fingerprint(model)
        self.assertEqual(original["elements"], 2)
        self.assertEqual(original, probe.model_weight_fingerprint(model))
        values[0, 1] = 3
        self.assertNotEqual(original["sha256"], probe.model_weight_fingerprint(model)["sha256"])
        values[0, 1] = 2
        parameter.name = "b"
        self.assertNotEqual(original["sha256"], probe.model_weight_fingerprint(model)["sha256"])
        parameter.name = "a"
        values.shape = (2, 1)
        self.assertNotEqual(original["sha256"], probe.model_weight_fingerprint(model)["sha256"])

    def test_production_profile_is_explicit_and_shape_checked(self):
        args = SimpleNamespace(gpu_training_profile="redesign-v1")
        config = SimpleNamespace(mamba2_faithful=True, mamba_d_state=64, mamba_head_dim=64)
        with patch.dict(probe.os.environ, {}, clear=True):
            policy = probe.core.configure_gpu_training_profile(args, config, True, 32)
            self.assertEqual(policy["switches"]["NSOS_MAMBA_BOUNDARY_HISTORY"], "1")
            self.assertEqual(probe.os.environ["NSOS_MAMBA_BACKWARD_CHUNK_SIZE"], "32")
            self.assertEqual(policy["ttt_derivative"], "truncated_stop_gradient_adaptation")
            with self.assertRaises(ValueError):
                probe.core.configure_gpu_training_profile(args, config, False, 32)
            config.mamba_head_dim = 4
            with self.assertRaises(ValueError):
                probe.core.configure_gpu_training_profile(args, config, True, 32)

    def test_kan_policy_is_explicit_and_native_verified(self):
        args = SimpleNamespace(gpu_training_profile="inherit", kan_compute_policy="tiled-v1")
        config = SimpleNamespace(use_kan=True, use_ttt=False)
        with patch.dict(probe.os.environ, {}, clear=True):
            policy = probe.core.configure_gpu_training_profile(args, config, True, 32)
            self.assertEqual(policy["switches"]["NSOS_KAN_RECOMPUTE_TRAINING"], "1")
            with self.assertRaisesRegex(RuntimeError, "kan.training_policy"):
                probe.core.validate_native_training_policy(policy, {})
            self.assertFalse(policy["native_verified"])
            probe.core.validate_native_training_policy(policy, {"kan.training_policy": "device_qat_tree256_rbf_tile16_recompute_v1"})
            self.assertTrue(policy["native_verified"])
            with self.assertRaises(ValueError):
                probe.core.configure_gpu_training_profile(args, config, False, 32)
            config.use_kan=False
            with self.assertRaises(ValueError):
                probe.core.configure_gpu_training_profile(args, config, True, 32)

    def test_kan_policy_rejects_malformed_inheritance(self):
        args = SimpleNamespace(gpu_training_profile="inherit")
        config = SimpleNamespace(use_kan=True, use_ttt=False)
        with patch.dict(probe.os.environ, {"NSOS_KAN_RECOMPUTE_TRAINING": "1invalid"}, clear=True):
            with self.assertRaises(ValueError):
                probe.core.configure_gpu_training_profile(args, config, True, 32)

    def test_kan_wmma_requires_native_identity_and_wave32(self):
        args = SimpleNamespace(gpu_training_profile="inherit", kan_compute_policy="wmma-v1")
        config = SimpleNamespace(use_kan=True, use_ttt=False)
        with patch.dict(probe.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "wave32"):
                probe.core.configure_gpu_training_profile(args, config, True, 64)
            policy = probe.core.configure_gpu_training_profile(args, config, True, 32)
            identity = {"kan.training_policy": policy["kan_compute"]}
            with self.assertRaisesRegex(RuntimeError, "kan.wmma_policy"):
                probe.core.validate_native_training_policy(policy, identity)
            self.assertFalse(policy["native_verified"])
            identity["kan.wmma_policy"] = policy["kan_wmma"]
            probe.core.validate_native_training_policy(policy, identity)
            self.assertTrue(policy["native_verified"])

    def test_kan_wmma_inheritance_requires_recompute(self):
        args = SimpleNamespace(gpu_training_profile="inherit")
        config = SimpleNamespace(use_kan=True, use_ttt=False)
        for value in ("1", "malformed"):
            with patch.dict(probe.os.environ, {"NSOS_KAN_WMMA_TRAINING": value}, clear=True):
                with self.assertRaises(ValueError):
                    probe.core.configure_gpu_training_profile(args, config, True, 32)


if __name__ == "__main__":
    unittest.main()
