import os
import sys
import unittest
import importlib.util
from pathlib import Path

_DLL_HANDLES = []


def _candidate_build_paths():
    root = Path(__file__).resolve().parents[1]
    nsos_root = root / "nsos"
    candidates = []
    env_build = os.environ.get("NSOS_BUILD_DIR")
    if env_build:
        candidates.append(Path(env_build))
    candidates.extend(
        [
            nsos_root / "build-ci",
            nsos_root / "build_cuda129",
            nsos_root / "build",
        ]
    )
    return candidates


def _bootstrap_paths():
    for candidate in _candidate_build_paths():
        candidate = candidate.resolve()
        for path in (candidate, candidate / "Release"):
            if path.exists():
                if hasattr(os, "add_dll_directory"):
                    _DLL_HANDLES.append(os.add_dll_directory(str(path)))
                sys.path.insert(0, str(path))


def _import_nsos_ext():
    _bootstrap_paths()
    try:
        import nsos_ext  # type: ignore
        return nsos_ext
    except ImportError:
        for candidate in _candidate_build_paths():
            candidate = candidate.resolve()
            for root in (candidate, candidate / "Release"):
                if not root.exists():
                    continue
                matches = sorted(root.glob("nsos_ext*.pyd"))
                if not matches:
                    continue
                spec = importlib.util.spec_from_file_location("nsos_ext", matches[0])
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["nsos_ext"] = module
                    spec.loader.exec_module(module)
                    return module
        raise


nsos_ext = _import_nsos_ext()


class TestIndustrialFeatures(unittest.TestCase):
    def test_tokenizer_special_tokens(self):
        tok = nsos_ext.Tokenizer()
        tok.add_special_tokens(["<think>", "</think>"])

        text = "Hello <think> logic </think>"
        ids = tok.encode(text)
        decoded = tok.decode(ids)

        self.assertIn("<think>", decoded)
        self.assertIn("</think>", decoded)

    def test_deterministic_tensor_random(self):
        nsos_ext.set_seed(1337)
        a = nsos_ext.Tensor.random([8, 8], nsos_ext.Device.CPU).numpy().copy()
        nsos_ext.set_seed(1337)
        b = nsos_ext.Tensor.random([8, 8], nsos_ext.Device.CPU).numpy().copy()
        self.assertTrue((a == b).all())

    def test_inference_engine_checkpoint_contract(self):
        engine = nsos_ext.InferenceEngine()
        cfg = nsos_ext.ModelConfig()
        cfg.num_layers = 1
        cfg.d_model = 16
        cfg.vocab_size = 32
        self.assertFalse(engine.load_model("__missing_checkpoint__.bin", cfg))

    def test_context_forward_contract(self):
        model = nsos_ext.JambaModel(1, 16, 10, nsos_ext.Device.CPU)
        ctx = nsos_ext.Context(4)
        out = model.forward_ids([1, 2, 3, 4], ctx)
        self.assertEqual(out.shape, [4, 10])


if __name__ == "__main__":
    unittest.main()
