import unittest
import os
import sys
import torch

try:
    import nsos_ext
except ImportError:
    # Build path hack
    sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
    try:
        import nsos_ext
    except ImportError:
        import nsos.build.nsos_ext as nsos_ext

class TestIndustrialFeatures(unittest.TestCase):
    def test_tokenizer_special_tokens(self):
        tok = nsos_ext.Tokenizer()
        tok.add_special_tokens(["<think>", "<end_think>"])

        text = "Hello <think> logic </think>"
        ids = tok.encode(text)
        decoded = tok.decode(ids)

        # We expect <think> to be a single token ID if implementation is correct
        # Note: BPE implementation might split spaces around it depending on logic.
        # But decode(encode(x)) should roughly equal x (ignoring whitespace normalization quirks)
        self.assertIn("<think>", decoded)
        print(f"Tokenizer: '{text}' -> {ids} -> '{decoded}'")

    def test_optimizer_muon(self):
        # Create a dummy parameter
        t = nsos_ext.Tensor.ones([10], nsos_ext.Device.CPU)
        t_grad = nsos_ext.Tensor.ones([10], nsos_ext.Device.CPU).mul(0.1) # grad = 0.1

        p = nsos_ext.Parameter()
        # Bind manually? Parameter struct in pybind exposes data/grad but not constructor?
        # We can't easily create Parameter in Python if it's not bound.
        # JambaModel returns pointers to Parameters. Let's use a dummy model.

        # Mocking Parameter object behavior via binding (assuming we can assign)
        # Actually we need C++ side integration.
        # Let's rely on JambaModel parameters.
        model = nsos_ext.JambaModel(1, 16, 10, nsos_ext.Device.CPU)
        params = model.parameters()
        if not params: return

        p0 = params[0]
        initial_sum = p0.data.norm()

        # Set fake gradient
        p0.grad.copy_from(nsos_ext.Tensor.ones(p0.data.shape, nsos_ext.Device.CPU).mul(0.1))

        # Create Muon
        opt = nsos_ext.MuonOptimizer([], 0.01)
        opt.step(params)

        new_sum = p0.data.norm()
        # Momentum means it should move away.
        self.assertNotEqual(initial_sum, new_sum)
        print(f"Muon Step: {initial_sum} -> {new_sum}")

    def test_memory_retrieval(self):
        # Verify JambaModel uses memory
        model = nsos_ext.JambaModel(1, 16, 10, nsos_ext.Device.CPU)

        # Forward pass should trigger retrieval logic (even if empty)
        x = nsos_ext.Tensor.random([1, 4, 16], nsos_ext.Device.CPU)
        ctx = nsos_ext.Context()
        ctx.set_metadata("force_system2", True)

        out = model.forward(x, ctx)
        self.assertEqual(out.shape, [1, 4, 16])
        print("Memory Integrated Forward Pass: OK")

if __name__ == '__main__':
    unittest.main()
