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

def test_parts_5_8():
    print("=== Running Part 5-8 Rigorous Validation ===")

    engine = pantheon.Engine()

    # Part 5: Physical
    # Photonic Loss (High Freq Penalty)
    # W: 2x2.
    # [0, 1]
    # [0, 1]
    # Impl iterates y < h-1, x < w-1. Only (0,0) processed.
    # Val(0)=0. Right(1)=1. Down(2)=0.
    # Penalty: |0-1| + |0-0| = 1.
    # Loss: 1 / 4 = 0.25.
    w_pho = [0.0, 1.0, 0.0, 1.0]
    loss_pho = engine.compute_photonic_loss(w_pho, 2, 2)
    print(f"Photonic Loss (Exp: 0.25): {loss_pho}")
    if abs(loss_pho - 0.25) > 1e-5:
        print("FAIL: Photonic loss mismatch")
        sys.exit(1)

    # Memristive Noise
    # Stuck-at-0 (High rate 1.0) -> All become 0.
    w_mem = [1.0, 1.0]
    w_noisy = engine.inject_memristive_noise(w_mem, 1.0)
    print(f"Memristive (Exp: [0, 0]): {w_noisy}")
    if w_noisy != [0.0, 0.0]:
        print("FAIL: Memristive stuck-at fault mismatch")
        sys.exit(1)

    # Part 6: Abstract
    # Functorial (L2)
    l_fun = engine.compute_functorial_loss([1.0], [2.0])
    print(f"Functorial Loss (Exp: 1.0): {l_fun}")
    if abs(l_fun - 1.0) > 1e-5:
        print("FAIL")
        sys.exit(1)

    # ToM (L2)
    l_tom = engine.compute_tom_loss([1.0], [2.0])
    print(f"ToM Loss (Exp: 1.0): {l_tom}")
    if abs(l_tom - 1.0) > 1e-5:
        print("FAIL")
        sys.exit(1)

    # Part 7: Social
    # Nash Regret (Cross Entropy)
    # Nash: [1.0, 0.0]. Student: [0.5, 0.5].
    # Regret: -1*log(0.5) - 0 = 0.693.
    l_nash = engine.compute_nash_regret([1.0, 0.0], [0.5, 0.5])
    print(f"Nash Regret (Exp: ~0.693): {l_nash}")
    if abs(l_nash - 0.693147) > 1e-4:
        print("FAIL: Nash regret mismatch")
        sys.exit(1)

    # Swarm Update
    # Curr: [0]. Best: [10]. Vel: [0].
    # Inertia(0.5)*0 + Social(1.0)*0.5*(10-0) = 5.0.
    # New Pos: 0 + 5 = 5.
    new_w = engine.update_swarm([0.0], [10.0], [0.0])
    print(f"Swarm Update (Exp: [5.0]): {new_w}")
    if abs(new_w[0] - 5.0) > 1e-5:
        print("FAIL: Swarm update mismatch")
        sys.exit(1)

    # Part 8: Integrity
    # PATE (Laplace Noise)
    # Votes: [100]. Epsilon: 1.0.
    noisy_votes = engine.compute_pate_aggregation([100], 1.0)
    print(f"PATE Vote (Exp: ~100): {noisy_votes[0]}")
    if abs(noisy_votes[0] - 100) > 20:
        print("FAIL: PATE noise outlier or error")
        sys.exit(1)

    # Neural ODE Adjoint
    # z=[1], grad=[1], dt=0.1.
    # grad_prev = grad * (1+dt) = 1.1.
    adj = engine.compute_ode_adjoint([1.0], [1.0], 0.1)
    print(f"Adjoint Grad (Exp: 1.1): {adj[0]}")
    if abs(adj[0] - 1.1) > 1e-5:
        print("FAIL: ODE Adjoint mismatch")
        sys.exit(1)

    print("=== Part 5-8 Tests Passed ===")

if __name__ == "__main__":
    test_parts_5_8()
