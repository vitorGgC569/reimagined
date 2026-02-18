import sys
import os
import math

# Add the build directory to python path
sys.path.append(".")
sys.path.append("build")

try:
    import pantheon
except ImportError:
    print("Could not import pantheon module.")
    sys.exit(1)

def test_part2_cognition():
    print("=== Running Part 2 Rigorous Validation (CoT, Symbolic) ===")

    engine = pantheon.Engine()

    # 1. Test Chunk-Wise Training (CoT)
    # Trace 1: [1, 2, 3, 4]
    # Trace 2: [1, 2, 4, 5]
    # Chunk Size 2.
    # Chunk 1 (0-2): [1, 2] vs [1, 2]. Diff [0, 0]. Loss 0.
    # Chunk 2 (2-4): [3, 4] vs [4, 5]. Diff [-1, -1]. Sq Diff [1, 1]. Mean 1.
    # Total Loss: (0 + 1) / 2 = 0.5.

    t1 = [1.0, 2.0, 3.0, 4.0]
    t2 = [1.0, 2.0, 4.0, 5.0]
    loss_cwt = engine.compute_chunk_wise_loss(t1, t2, 2)
    print(f"CWT Loss (Exp: 0.5): {loss_cwt}")
    if abs(loss_cwt - 0.5) > 1e-5:
        print("FAIL: CWT Loss mismatch")
        sys.exit(1)

    # 2. Test Granularity Adjustment (MiCoTA)
    # Steps: [0, 1, 2, 3, 4, 5]
    # Competence: 0.0 (Novice). Stride = 1 + floor(0) = 1. Keep all.
    # Competence: 1.0 (Expert). Stride = 1 + floor(3) = 4. Keep 0, 4, and last (5).
    # Wait, implementation logic:
    # Stride = 1 + int(comp * 3).
    # Loop i: keep if i == last OR i % stride == 0.

    steps = [0, 1, 2, 3, 4, 5]

    # Novice
    adj_novice = engine.adjust_granularity(steps, 0.0)
    print(f"Novice (Exp: [0,1,2,3,4,5]): {adj_novice}")
    if len(adj_novice) != 6:
        print("FAIL: Novice granularity mismatch")
        sys.exit(1)

    # Expert
    adj_expert = engine.adjust_granularity(steps, 1.0) # Stride 4
    # i=0: Keep (0%4==0).
    # i=1,2,3: Skip.
    # i=4: Keep (4%4==0).
    # i=5: Keep (last).
    # Result: [0, 4, 5].
    print(f"Expert (Exp: [0,4,5]): {adj_expert}")
    if adj_expert != [0, 4, 5]:
        print("FAIL: Expert granularity mismatch")
        sys.exit(1)

    # 3. Test Symbolic Verification
    # Rule: Sum of vector must be X.
    # Vector: [1, 2, 3] -> Sum 6.
    # Expected: 6. Valid -> Loss 0.
    # Expected: 5. Invalid -> Loss 10.0 (default penalty).

    vec = [1.0, 2.0, 3.0]
    loss_valid = engine.compute_symbolic_loss(vec, 6.0)
    loss_invalid = engine.compute_symbolic_loss(vec, 5.0)

    print(f"Symbolic Loss Valid (Exp: 0.0): {loss_valid}")
    print(f"Symbolic Loss Invalid (Exp: 10.0): {loss_invalid}")

    if loss_valid != 0.0 or loss_invalid != 10.0:
        print("FAIL: Symbolic verification mismatch")
        sys.exit(1)

    print("=== Part 2 Tests Passed ===")

if __name__ == "__main__":
    test_part2_cognition()
