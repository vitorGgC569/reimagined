import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
import nsos_ext

class TestDeterminism(unittest.TestCase):
    def test_seed_control(self):
        """Determinism: Same Seed = Same Weights"""
        nsos_ext.set_seed(42)
        t1 = nsos_ext.Tensor.random([10, 10], nsos_ext.Device.CPU).numpy()

        nsos_ext.set_seed(42)
        t2 = nsos_ext.Tensor.random([10, 10], nsos_ext.Device.CPU).numpy()

        self.assertTrue(np.array_equal(t1, t2), "RNG is not deterministic")

    def test_model_determinism(self):
        """Determinism: Full Model Forward"""
        config = (2, 32, 64) # Layers, Dim, Vocab

        nsos_ext.set_seed(123)
        model1 = nsos_ext.JambaModel(*config, nsos_ext.Device.CPU)
        input_ids = [1, 5, 9]
        out1 = model1.forward_ids(input_ids).numpy()

        nsos_ext.set_seed(123)
        model2 = nsos_ext.JambaModel(*config, nsos_ext.Device.CPU)
        out2 = model2.forward_ids(input_ids).numpy()

        self.assertTrue(np.allclose(out1, out2), "Model inference is not deterministic")

if __name__ == '__main__':
    unittest.main()
