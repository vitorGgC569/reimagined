
import os
# Force Single Threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import sys
import unittest

# Try to import nsos_ext
try:
    import nsos_ext
except ImportError:
    nsos_ext = None

class TestRegression(unittest.TestCase):
    def setUp(self):
        if nsos_ext is None:
            self.skipTest("NSOS Extension not found")

        self.golden_input_path = "tests/data/golden_input.pt"
        self.golden_output_path = "tests/data/golden_output_v1.pt"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_golden_record_integrity(self):
        """
        Verify that the model output is EXACTLY identical to the recorded 'Golden' version.
        This detects subtle numerical regressions caused by refactors.
        """
        if not os.path.exists(self.golden_input_path) or not os.path.exists(self.golden_output_path):
            self.skipTest("Golden Records not found. Run generate_golden.py first.")
            return

        # 1. Determinism
        nsos_ext.set_seed(42)
        torch.manual_seed(42)

        # 2. Load Data
        x_torch = torch.load(self.golden_input_path)
        y_expected = torch.load(self.golden_output_path)

        # 3. Setup Model (Must match Generation Config exactly)
        D = x_torch.shape[-1]
        model = nsos_ext.JambaModel(2, D, 100, nsos_ext.Device.CPU) # Force CPU for golden consistency
        # Assuming weights are initialized deterministically by set_seed(42) inside constructor?
        # If JambaModel random init depends on global RNG which we seeded, it should be same.

        # 4. Forward
        # Map torch -> nsos
        x_cpu = x_torch.cpu().float().contiguous()
        t_in = nsos_ext.Tensor(list(x_cpu.shape), nsos_ext.Device.CPU)
        t_in.copy_from(nsos_ext.Tensor.from_blob(x_cpu.data_ptr(), list(x_cpu.shape), nsos_ext.Device.CPU))

        out_nsos = model.forward(t_in)

        # 5. Compare
        out_torch = torch.from_numpy(out_nsos.numpy())

        # Strict Tolerance for Regression
        # If float32, usually 1e-6.
        # But if different CPU/AVX instruction set?
        # Golden Record is usually generated on CI machine.
        # Here we assume local consistency.

        # Calculate Max Diff
        diff = (out_torch - y_expected.cpu()).abs().max().item()

        self.assertTrue(diff < 1e-5, f"Regression Detected! Max Diff: {diff}")
        print(f"✅ Regression Test Passed. Diff: {diff:.9f}")

if __name__ == '__main__':
    unittest.main()
