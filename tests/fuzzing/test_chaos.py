import sys
import os
import unittest
import random
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
try:
    import nsos_ext
except ImportError:
    pass

class Fuzzer:
    def __init__(self):
        self.device = nsos_ext.Device.CPU

    def fuzz_shape(self):
        # Generate weird shapes
        rank = random.randint(0, 5)
        if rank == 0: return []
        shape = []
        for _ in range(rank):
            dim = random.choice([0, 1, 13, 1024, -1])
            shape.append(dim)
        return shape

    def run(self, iterations=100):
        print(f"=== 🌪️ CHAOS FUZZING (N={iterations}) ===")
        crashes = 0
        caught = 0

        for i in range(iterations):
            try:
                # 1. Tensor Fuzz
                shape = self.fuzz_shape()
                # print(f"Fuzzing Tensor({shape})...", end="\r")

                # Should throw RuntimeError, not Segfault
                t = nsos_ext.Tensor.zeros(shape, self.device)

                # 2. Ops Fuzz
                if t.size > 0:
                    t2 = t.add(t)

            except Exception as e:
                # print(f"Caught expected error: {e}")
                caught += 1
            except:
                print("CRITICAL: Python Crash (Segfault-like?)")
                crashes += 1

        print(f"\nResult: {caught} Caught Exceptions, {crashes} Hard Crashes.")
        if crashes > 0:
            print("❌ TEST FAILED: System is fragile.")
        else:
            print("✅ TEST PASSED: System is resilient.")

if __name__ == "__main__":
    if 'nsos_ext' in sys.modules:
        Fuzzer().run()
    else:
        print("Skipping Fuzzing (No Lib)")
