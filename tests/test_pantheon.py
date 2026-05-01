import sys
import os
import time

# Add the build directory to python path to find the module
sys.path.append(".")
sys.path.append("build")

try:
    import pantheon
except ImportError:
    print("Could not import pantheon module. Make sure it is built and in the path.")
    sys.exit(1)

def test_integration():
    print("Testing Pantheon <-> KernelOpen Integration...")

    engine = pantheon.Engine()
    engine.initialize()

    # Test KernelOpen
    engine.submit_dummy_task()
    time.sleep(1)
    t = engine.get_throughput()
    print(f"Kernel Throughput: {t}")
    if t <= 0:
        print("FAIL: Kernel simulation failed")
        sys.exit(1)

    # Test Gradient Matching
    print("Testing Gradient Matching...")
    g1 = [1.0, 2.0, 3.0]
    g2 = [1.1, 2.1, 3.1]
    # Diff is 0.1, 0.1, 0.1
    # Sq Diff is 0.01, 0.01, 0.01
    # Sum Sq is 0.03
    loss = engine.compute_gradient_loss(g1, g2)
    print(f"Gradient Loss: {loss}")

    if abs(loss - 0.03) > 0.0001:
        print(f"FAIL: Expected ~0.03, got {loss}")
        sys.exit(1)

    print("Shutting down...")
    engine.shutdown()
    print("Test Passed!")

if __name__ == "__main__":
    test_integration()
