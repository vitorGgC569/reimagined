import sys
import os
import unittest
import numpy as np
from stats_utils import StatsTracker, AcceptanceCriteria

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
import nsos_ext

class TestGradient(unittest.TestCase):
    def test_gradient_stats(self):
        """Gradient Check: Analytical vs Numerical (Statistical)"""
        print("\n--- Gradient Accuracy Statistics ---")
        tracker = StatsTracker("Gradient Gap")
        ac = AcceptanceCriteria()

        # Test across range of inputs
        x_vals = np.linspace(-10, 10, 20)

        for x_val in x_vals:
            # f(x) = 0.5 * x^2 -> f'(x) = x (Assuming MSE is sum(sq)/N)
            # Tensor::mse_loss uses "2.0 * diff / size".
            # diff = x - 0 = x.
            # grad = 2.0 * x / 1 = 2x.

            x = nsos_ext.Tensor([1, 1], nsos_ext.Device.CPU)
            x.numpy()[:] = x_val

            target = nsos_ext.Tensor.zeros([1, 1], nsos_ext.Device.CPU)
            _, grad = x.mse_loss(target)

            # Fix: Use .item() to extract scalar from 0-d or 1-d array
            calc_grad = grad.numpy().item()
            expected_grad = 2.0 * x_val

            error = float(abs(calc_grad - expected_grad))
            tracker.add(error)

        avg_err = tracker.report()
        if avg_err is not None:
            self.assertTrue(ac.check("Avg Gradient Error", avg_err, 1e-4, "<"))

if __name__ == '__main__':
    unittest.main()
