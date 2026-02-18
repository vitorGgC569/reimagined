import time
import sys
import unittest

# Add project root to path
sys.path.append(".")

from sdk.uhk.bci import BCIController
from sdk.uhk.quantum import QuantumBridge
import uhk.uhk_core as uhk_core

class TestExoticHardware(unittest.TestCase):

    def test_bci_synthetic_stream(self):
        print("\n--- Testing BCI (Synthetic BrainFlow) ---")
        controller = BCIController(use_synthetic=True)
        controller.start()

        # Let it stream for 2 seconds
        time.sleep(2)

        controller.stop()
        print("BCI Test Passed.")

    def test_quantum_simulation(self):
        print("\n--- Testing Quantum Bridge (Qiskit Aer) ---")
        bridge = QuantumBridge(api_key=None)
        result = bridge.run_bell_state()

        # Bell state should have roughly 50% 00 and 50% 11
        # e.g. {'00': 500, '11': 524}
        self.assertTrue('00' in result or '0000' in result) # Check for key presence
        print("Quantum Test Passed.")

    def test_cpp_bindings_direct(self):
        print("\n--- Testing Direct C++ Bindings ---")

        # Ghost Memory
        mem = uhk_core.HolographicMemory(64)
        pat = mem.encode([0.1, 0.5, -0.2])
        mem.store("memory_alpha", pat)
        recall = mem.recall(pat)
        print(f"Ghost Recall: {recall}")
        self.assertEqual(recall, "memory_alpha")

        # Photonic
        eng = uhk_core.InterferenceEngine(4)
        res = eng.compute([complex(1,0)] * 4)
        print(f"Photonic Result: {res}")
        self.assertEqual(len(res), 4)

if __name__ == "__main__":
    unittest.main()
