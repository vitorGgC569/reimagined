import os
import sys
import unittest
from pathlib import Path

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


def _import_nsos_ext():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nsos" / "scripts"))
    from native_module import load_native_module, resolve_native_build_dir
    explicit = os.environ.get("NSOS_BUILD_DIR")
    directory = resolve_native_build_dir(Path(explicit) if explicit else None, _candidate_build_paths())
    module = load_native_module(directory)
    print(f"Native artifact: {module.__file__}")
    return module


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
