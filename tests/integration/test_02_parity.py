import sys
import os
import unittest
import numpy as np
from stats_utils import StatsTracker, AcceptanceCriteria

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
import nsos_ext

class TestParity(unittest.TestCase):
    def test_matmul_parity_stats(self):
        """Parity: MatMul vs NumPy (Statistical)"""
        print("\n--- MatMul Parity Statistics ---")
        tracker = StatsTracker("Error Magnitude")
        ac = AcceptanceCriteria()

        shapes = [(16,16), (32,32), (64,64), (128,128)]

        for shape in shapes:
            np_a = np.random.randn(*shape).astype(np.float32)
            np_b = np.random.randn(*shape).astype(np.float32)

            # Fix: Remove 4th argument (ownership is implicit false in current binding lambda)
            t_a = nsos_ext.Tensor.from_blob(np_a.ctypes.data, list(shape), nsos_ext.Device.CPU)
            t_b = nsos_ext.Tensor.from_blob(np_b.ctypes.data, list(shape), nsos_ext.Device.CPU)

            t_c = t_a.matmul(t_b)
            res_c = t_c.numpy()
            np_c = np.matmul(np_a, np_b)

            diff = np.abs(res_c - np_c)
            max_err = float(np.max(diff)) # Convert to float
            mean_err = float(np.mean(diff))
            tracker.add(max_err)

            print(f"Shape {shape}: MaxErr={max_err:.2e}, MeanErr={mean_err:.2e}")

        mean_max_err = tracker.report()
        if mean_max_err is not None:
            self.assertTrue(ac.check("Mean Max Error", mean_max_err, 1e-3, "<")) # Relaxed tolerance for float32

if __name__ == '__main__':
    unittest.main()
