import sys
import os
import unittest

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
try:
    import nsos_ext
except ImportError:
    print("CRITICAL: nsos_ext not found. Tests cannot run.")
    sys.exit(1)

class TestBindings(unittest.TestCase):
    def test_import(self):
        """Smoke Test: Does it import?"""
        self.assertIsNotNone(nsos_ext)

    def test_tensor_instantiation(self):
        """Can we create a Tensor?"""
        t = nsos_ext.Tensor([4, 4], nsos_ext.Device.CPU)
        self.assertEqual(t.shape, [4, 4])

    def test_model_instantiation(self):
        """Can we create the Model?"""
        model = nsos_ext.JambaModel(2, 64, 128, nsos_ext.Device.CPU)
        self.assertIsNotNone(model)

    def test_error_handling(self):
        """Do we get readable errors instead of segfaults?"""
        # Example: Invalid shape for matrix mult
        a = nsos_ext.Tensor([2, 3], nsos_ext.Device.CPU)
        b = nsos_ext.Tensor([4, 5], nsos_ext.Device.CPU) # 3 != 4
        with self.assertRaises(RuntimeError):
            a.matmul(b)

if __name__ == '__main__':
    unittest.main()
