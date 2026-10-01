"""Offline regression tests for release/evaluation/deployment contracts."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import container_runtime
import native_module
import run_scorecard as cli
from eval.adapters.dummy_adapter import DummyAdapter
from eval.adapters.nsos_adapter import NsosAdapter
from eval.benchmarks.base import BenchmarkResult
from eval.orchestrator import ScorecardResult, run_scorecard


class CoreRuntimeContracts(unittest.TestCase):
    def test_native_artifact_fingerprint_covers_real_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "fixture.pyd"
            binary.write_bytes(b"native-test-fixture")
            identity = native_module.native_artifact_identity(SimpleNamespace(__file__=str(binary)))
            self.assertEqual(identity["path"], str(binary.resolve()))
            self.assertEqual(identity["sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())

    def test_explicit_native_build_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ("nsos_ext" + native_module.EXTENSION_SUFFIXES[0])).touch()
            with self.assertRaisesRegex(RuntimeError, "No compatible"):
                native_module.resolve_native_build_dir(root / "missing", [root])
            self.assertEqual(native_module.resolve_native_build_dir(root), root.resolve())
            self.assertEqual(native_module.resolve_native_build_dir(None, [root]), root.resolve())

    def test_cached_native_from_other_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ("nsos_ext" + native_module.EXTENSION_SUFFIXES[0])).touch()
            cached = SimpleNamespace(__file__=str(root / "other" / "nsos_ext.pyd"))
            with mock.patch.dict(sys.modules, {"nsos_ext": cached}):
                with self.assertRaisesRegex(RuntimeError, "different build"):
                    native_module.load_native_module(root)

    def test_native_extension_uses_platform_abi_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".cpython-312-x86_64-linux-gnu.so", ".cp312-win_amd64.pyd"):
                with self.subTest(suffix=suffix), mock.patch.object(native_module, "EXTENSION_SUFFIXES", [suffix]):
                    self.assertFalse(native_module.has_native_module(root))
                    binary = root / ("nsos_ext" + suffix)
                    binary.touch()
                    self.assertTrue(native_module.has_native_module(root))
                    binary.unlink()
            (root / "nsos_ext.fake").touch()
            self.assertFalse(native_module.has_native_module(root))

    def test_scorecard_fails_closed(self):
        result = ScorecardResult("test", "fixture", 1)
        self.assertTrue(cli.scorecard_failures(result))
        benchmark = BenchmarkResult("fixture", "accuracy", {"accuracy": 0.5}, 2, 0.01)
        result.benchmarks = [benchmark]
        self.assertFalse(cli.scorecard_failures(result))
        for status in ("skipped", "partial", "error"):
            benchmark.status = status
            self.assertTrue(cli.scorecard_failures(result))
        benchmark.status = "ok"
        for value in (None, True, float("nan"), float("inf"), "invalid"):
            benchmark.metrics["accuracy"] = value
            self.assertTrue(cli.scorecard_failures(result))
        benchmark.metrics = {"accuracy": 0.5}
        benchmark.n_examples = 0
        self.assertTrue(cli.scorecard_failures(result))

    def test_empty_scorecard_selection_is_not_default_suite(self):
        for options in ({"benchmarks": []}, {"seeds": []}):
            with self.assertRaises(ValueError):
                run_scorecard(DummyAdapter(), **options)
        for args in (["--only"], ["--seeds"]):
            with self.assertRaises(SystemExit) as caught:
                cli.main(["--adapter", "dummy", *args])
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(cli.main(["--adapter", "dummy", "--only", "typo"]), 2)

    def test_multiseed_does_not_hide_a_failed_or_nonfinite_seed(self):
        for bad in ("skipped", "nan"):
            def runner(adapter, seed=0):
                status = "skipped" if seed == 2 and bad == "skipped" else "ok"
                value = float("nan") if seed == 2 and bad == "nan" else 0.5
                return BenchmarkResult("fixture", "accuracy", {"accuracy": value}, 2, 0.01, status=status)
            result = run_scorecard(DummyAdapter(), benchmarks=[("fixture", runner, {}, True)], seeds=[1, 2])
            self.assertEqual(result.benchmarks[0].status, "error")
            self.assertTrue(cli.scorecard_failures(result))

    def test_container_requires_model_token_and_exact_proxy_peers(self):
        with tempfile.TemporaryDirectory() as model:
            env = {"NSOS_MODEL": model, "NSOS_API_TOKEN": "a-long-test-token", "NSOS_TRUSTED_PROXY_IPS": "10.0.0.2,::1"}
            argv = container_runtime.server_arguments(env)
            self.assertIn("--require-tls-proxy-header", argv)
            self.assertIn("--trust-proxy-headers", argv)
            self.assertNotIn(env["NSOS_API_TOKEN"], argv)
            for key, value in (("NSOS_MODEL", ""), ("NSOS_API_TOKEN", "short"),
                               ("NSOS_TRUSTED_PROXY_IPS", ""), ("NSOS_TRUSTED_PROXY_IPS", "10.0.0.0/8"),
                               ("NSOS_PORT", "70000")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    container_runtime.server_arguments({**env, key: value})

    def test_readiness_checks_real_model_state_and_disables_proxy(self):
        for ready in (True, False):
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.status = 200
            response.read.return_value = json.dumps({"status": "ready" if ready else "starting", "ok": ready}).encode()
            opener = mock.Mock()
            opener.open.return_value = response
            with mock.patch.object(container_runtime.urllib.request, "build_opener", return_value=opener) as build:
                if ready:
                    container_runtime.check_ready({"NSOS_API_TOKEN": "test-token"})
                else:
                    with self.assertRaises(RuntimeError):
                        container_runtime.check_ready({"NSOS_API_TOKEN": "test-token"})
                self.assertEqual(build.call_args.args[0].proxies, {})
                request = opener.open.call_args.args[0]
                self.assertEqual(request.full_url, "http://127.0.0.1:8080/ready")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-token")

    def fake_native(self):
        engine = mock.Mock()
        engine.load_model.return_value = True
        engine.model_config.return_value = SimpleNamespace(max_context_tokens=64, vocab_size=256)
        engine.parameter_count.return_value = 123
        engine.tokenize.return_value = [2]
        native = SimpleNamespace(__file__=__file__, InferenceEngine=lambda: engine, ModelLoadOptions=SimpleNamespace,
            ModelConfig=lambda: SimpleNamespace(vocab_size=256, max_context_tokens=64, use_cuda=False),
            Device=SimpleNamespace(CPU="cpu", GPU="gpu"), GenerationOptions=SimpleNamespace)
        return engine, native

    def test_pack_evaluation_delegates_to_strict_sdk(self):
        engine, native = self.fake_native()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(NsosAdapter, "_import_nsos", return_value=native):
            (Path(directory) / "manifest.nsos").write_text("strict-fixture", encoding="utf-8")
            adapter = NsosAdapter(directory, directory, device="cpu")
            self.assertEqual(engine.load_model.call_count, 1)
            engine.load_tokenizer.assert_not_called()
            engine.generate_ex.return_value = "answer STOP ignored"
            self.assertEqual(adapter.generate("prompt", stop_sequences=["STOP"]), "answer ")
            self.assertEqual(adapter.capability.max_seq_len, 64)
            self.assertIn("SDK strict load; sha256=", adapter.capability.notes)
            with self.assertRaisesRegex(ValueError, "overrides are forbidden"):
                NsosAdapter(directory, directory, device="cpu", tokenizer_path="replacement")

    def test_raw_evaluation_rejects_unknown_config_and_failed_load(self):
        engine, native = self.fake_native()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(NsosAdapter, "_import_nsos", return_value=native):
            root = Path(directory)
            checkpoint = root / "weights.bin"
            checkpoint.touch()
            config = root / "effective_model_config.json"
            config.write_text('{"misspelled_architecture": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported model configuration"):
                NsosAdapter(checkpoint, root, device="cpu")
            engine.load_model.assert_not_called()
            config.write_text('{"vocab_size": 256}', encoding="utf-8")
            engine.load_model.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Strict model load failed"):
                NsosAdapter(checkpoint, root, device="cpu")
            self.assertEqual(engine.load_model.call_count, 1)
            engine.load_tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
