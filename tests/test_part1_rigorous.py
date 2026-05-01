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

def test_part1_rigorous():
    print("=== Running Part 1 Rigorous Validation (MSDCRD, NTCE, ULD, CAM) ===")

    engine = pantheon.Engine()

    # 1. Test MSDCRD (Multi-Scale Pooling)
    # Tensor: 1x1x4x4 (Batch, Channel, Height, Width)
    # We need patches that are distinct (orthogonal preferred) to minimize negative similarity.
    # Patch Size 2x2. Grid 2x2.
    # P1: [1, 0, 0, 0]
    # P2: [0, 1, 0, 0]
    # P3: [0, 0, 1, 0]
    # P4: [0, 0, 0, 1]
    # These are perfectly orthogonal. Sim(i, j) = 0 if i!=j.
    # Sim(i, i) = 1.
    # Numerator: exp(1/0.07) ~ exp(14.28)
    # Denom: exp(14.28) + 3*exp(0) = exp(14.28) + 3
    # Loss: -log(exp(14.28) / (exp(14.28)+3)) ~ -log(1) ~ 0.

    # 4x4 Tensor (Row Major)
    # Row 0: 1 0 | 0 1
    # Row 1: 0 0 | 0 0
    # -----------+-----
    # Row 2: 0 0 | 0 0
    # Row 3: 1 0 | 0 1  <- Wait, let's make P3/P4 distinct.

    # Let's define patches directly in row major 4x4.
    # P1 (TL): 1 0 / 0 0
    # P2 (TR): 0 1 / 0 0
    # P3 (BL): 0 0 / 1 0
    # P4 (BR): 0 0 / 0 1

    s_feat = [
        1.0, 0.0,   0.0, 1.0,
        0.0, 0.0,   0.0, 0.0,

        0.0, 0.0,   0.0, 0.0,
        1.0, 0.0,   0.0, 1.0
    ]
    t_feat = s_feat

    loss_msdcrd = engine.compute_msdcrd_loss(s_feat, t_feat, 1, 1, 4, 4, 2, 2)
    print(f"MSDCRD Loss (Exp: ~0.0): {loss_msdcrd}")
    if loss_msdcrd > 0.01:
        print("FAIL: MSDCRD Loss too high")
        sys.exit(1)

    # 2. Test NTCE-KD (Logits)
    # Target: Class 0
    # Teacher: [0.9, 0.1]
    # Student: [0.9, 0.1]
    # Loss = -1 * 0.69 * log(0.69) - 2.0 * 0.31 * log(0.31) = 0.97
    s_logits = [0.9, 0.1]
    t_logits = [0.9, 0.1]
    loss_ntce = engine.compute_ntce_loss(s_logits, t_logits, 0, 2.0)
    print(f"NTCE Loss (Self-Distillation): {loss_ntce}")
    if loss_ntce <= 0:
        print("FAIL: NTCE Loss invalid")
        sys.exit(1)

    # 3. Test Optimal Transport (Sinkhorn)
    # Move mass from class 0 to class 1. Cost 1.0.
    s_ot = [10.0, -10.0]
    t_ot = [-10.0, 10.0]
    loss_ot = engine.compute_optimal_transport_loss(s_ot, t_ot)
    print(f"OT Loss (Exp: ~1.0): {loss_ot}")
    if abs(loss_ot - 1.0) > 0.2:
        print(f"FAIL: OT Loss mismatch {loss_ot}")
        sys.exit(1)

    # 4. Test CAM (Attention)
    w = [0.5, 0.5]
    f = [1.0, 2.0, 3.0, 4.0]
    loss_cam = engine.compute_cam_loss(f, w, f, w, 1, 1, 2, 2, 2, 0)
    print(f"CAM Loss (Exp: 0.0): {loss_cam}")
    if loss_cam > 1e-5:
        print("FAIL: CAM Loss mismatch")
        sys.exit(1)

    print("=== Part 1 Tests Passed ===")

if __name__ == "__main__":
    test_part1_rigorous()
