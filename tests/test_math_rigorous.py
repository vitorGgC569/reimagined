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

def test_math_rigorous():
    print("=== Running Rigorous Math Validation ===")

    engine = pantheon.Engine()
    # We don't need to initialize KernelOpen for pure math tests,
    # but the class might expect it or it's fine.
    # The math methods are stateless wrappers in the current implementation.

    # 1. Test Gradient Matching (L2 Loss)
    # Vectors: [1, 2], [4, 6]
    # Diff: [3, 4]
    # Sq Sum: 9 + 16 = 25
    g1 = [1.0, 2.0]
    g2 = [4.0, 6.0]
    loss_gkd = engine.compute_gradient_loss(g2, g1)
    print(f"GKD Loss (Exp: 25.0): {loss_gkd}")
    if abs(loss_gkd - 25.0) > 1e-5:
        print("FAIL: GKD Loss mismatch")
        sys.exit(1)

    # 2. Test Relational Distillation (Distance-wise)
    # Batch size 2, Dim 1
    # Student: [1], [2] -> Dist Matrix: [[0, 1], [1, 0]] -> Mean: 0.5
    # Teacher: [10], [20] -> Dist Matrix: [[0, 10], [10, 0]] -> Mean: 5.0
    # Normalized S: [[0, 2], [2, 0]]
    # Normalized T: [[0, 2], [2, 0]]
    # Diff: 0
    s_rkd = [1.0, 2.0]
    t_rkd = [10.0, 20.0]
    loss_rkd = engine.compute_relational_loss(s_rkd, t_rkd, 2, 1)
    print(f"RKD Loss (Exp: 0.0): {loss_rkd}")
    if abs(loss_rkd - 0.0) > 1e-5:
        print("FAIL: RKD Loss mismatch")
        sys.exit(1)

    # 3. Test Contrastive Distillation (CRD)
    # Batch size 2, Dim 2
    # Student: A=[1, 0], B=[0, 1]
    # Teacher: A=[1, 0], B=[0, 1] (Perfect alignment)
    # Cosine Sim:
    # (A, A) = 1.0
    # (A, B) = 0.0
    # Numerator (Pos): exp(1.0 / 0.07) ~ exp(14.28)
    # Denominator: exp(14.28) + exp(0) ~ exp(14.28) + 1
    # Loss term: -log(exp(14.28) / (exp(14.28) + 1)) ~ -log(1) ~ 0
    # Ideally loss should be very small.
    s_crd = [1.0, 0.0, 0.0, 1.0]
    t_crd = [1.0, 0.0, 0.0, 1.0]
    loss_crd = engine.compute_contrastive_loss(s_crd, t_crd, 2, 2)
    print(f"CRD Loss (Perfect Match) (Exp: ~0.0): {loss_crd}")
    if loss_crd > 0.01:
        print("FAIL: CRD Loss too high for perfect match")
        sys.exit(1)

    print("=== All Rigorous Math Tests Passed ===")

if __name__ == "__main__":
    test_math_rigorous()
