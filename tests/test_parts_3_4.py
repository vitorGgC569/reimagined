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

def test_parts_3_4():
    print("=== Running Part 3 & 4 Rigorous Validation ===")

    engine = pantheon.Engine()

    # Part 3: Physics (IB, Topology)

    # 1. Information Bottleneck (Compression)
    # Means: [0, 0]. LogVars: [0, 0] (Var=1).
    # KL(N(0,1)||N(0,1)) should be 0.
    loss_ib_zero = engine.compute_ib_loss([0.0, 0.0], [0.0, 0.0])
    print(f"IB Loss Zero (Exp: 0.0): {loss_ib_zero}")
    if abs(loss_ib_zero) > 1e-5:
        print("FAIL: IB Loss mismatch")
        sys.exit(1)

    # Means: [1]. LogVars: [0]. Var=1.
    # KL = -0.5 * (1 + 0 - 1 - 1) = -0.5 * (-1) = 0.5.
    loss_ib_val = engine.compute_ib_loss([1.0], [0.0])
    print(f"IB Loss Val (Exp: 0.5): {loss_ib_val}")
    if abs(loss_ib_val - 0.5) > 1e-5:
        print("FAIL: IB Loss val mismatch")
        sys.exit(1)

    # 2. Topology (Persistence Barcode)
    # 3 points in line: 0 --1-- 1 --1-- 2
    # Distances: (0,1)=1, (1,2)=1, (0,2)=2.
    # MST Edges: (0,1) w=1, (1,2) w=1.
    # Components: 3 -> merge(1) -> 2 -> merge(1) -> 1.
    # Barcodes: Death at 1.0, Death at 1.0. (One persists forever).
    # If Student == Teacher, Loss 0.

    # Let's test non-zero loss.
    # S: Dist [1, 1, 2] -> Deaths [1, 1]
    # T: Dist [2, 2, 4] -> Deaths [2, 2] (Scaled by 2)
    # Diff: (1-2)^2 + (1-2)^2 = 1 + 1 = 2.

    s_dist = [0, 1, 2, 1, 0, 1, 2, 1, 0] # 3x3 matrix flattened.
    t_dist = [0, 2, 4, 2, 0, 2, 4, 2, 0]
    loss_topo = engine.compute_topology_loss(s_dist, t_dist, 3)
    print(f"Topology Loss (Exp: 2.0): {loss_topo}")
    if abs(loss_topo - 2.0) > 1e-5:
        print("FAIL: Topology Loss mismatch")
        sys.exit(1)

    # Part 4: Frontier (Meta, Causal, Quantum, Spiking)

    # 1. Meta-Distiller
    # Current T=1.0. Student Loss=0.5. Target=0.1. Error=0.4.
    # New T = 1.0 + 0.01 * 0.4 = 1.004.
    new_temp = engine.update_meta_policy(1.0, 0.5)
    print(f"Meta Temp (Exp: 1.004): {new_temp}")
    if abs(new_temp - 1.004) > 1e-5:
        print("FAIL: Meta update mismatch")
        sys.exit(1)

    # 2. Causal Invariance
    # Orig: [1, 2]. Interv: [1, 3]. Diff: [0, -1]. Sq: 1.
    loss_causal = engine.compute_causal_loss([1.0, 2.0], [1.0, 3.0])
    print(f"Causal Loss (Exp: 1.0): {loss_causal}")
    if abs(loss_causal - 1.0) > 1e-5:
        print("FAIL: Causal loss mismatch")
        sys.exit(1)

    # 3. Quantum Fidelity
    # State S: [1, 0] (Real). |psi> = 1 + 0i.
    # State T: [0, 1] (Imag). |phi> = 0 + 1i.
    # Dot: (1)(0) + (0)(1) = 0 (Real part).
    # Cross: (1)(1) - (0)(0) = 1 (Imag part).
    # Inner prod: 0 + 1i. Mag Sq: 1.
    # Loss: 1 - 1 = 0.
    # Wait. <1|i> = -i. |<-i>|^2 = 1.
    # They are orthogonal in real space? No, complex plane.
    # [1, 0] is real axis unit vector. [0, 1] is imag axis unit vector.
    # They represent the SAME quantum state phase-shifted? No.
    # Fidelity measures "closeness".
    # Let's try Orthogonal states.
    # S: [1, 0] (Real). |0>
    # T: [0, 0] (Zero? No states must be norm 1).
    # T: [0, 0, 1, 0] (If 2D complex space).
    # My impl assumes 1D complex stream?
    # Impl: sum over pairs.
    # Let's use 2 complex numbers (4 floats).
    # S: [1,0, 0,0] -> |1, 0>
    # T: [0,0, 1,0] -> |0, 1>
    # Dot Real: 1*0 + ... = 0.
    # Dot Imag: 0.
    # Fidelity 0. Loss 1.

    s_q = [1.0, 0.0, 0.0, 0.0]
    t_q = [0.0, 0.0, 1.0, 0.0]
    loss_q = engine.compute_quantum_loss(s_q, t_q)
    print(f"Quantum Loss (Orthogonal) (Exp: 1.0): {loss_q}")
    if abs(loss_q - 1.0) > 1e-5:
        print("FAIL: Quantum loss mismatch")
        sys.exit(1)

    # 4. Spiking Distance
    # S: [1, 0] -> Filt: [1, exp(-1/tau)]
    # T: [0, 1] -> Filt: [0, 1] (at t=1). At t=0? 0.
    # My impl: val = val*decay + spike.
    # S: i=0: val=1. i=1: val=exp.
    # T: i=0: val=0. i=1: val=1.
    # Diff:
    # i=0: 1 - 0 = 1. Sq=1.
    # i=1: exp(-1/5) - 1 = 0.818 - 1 = -0.181. Sq=0.032.
    # Total: 1.032.

    tau = 5.0
    decay = math.exp(-1.0/tau) # 0.8187
    exp_loss = 1.0**2 + (decay - 1.0)**2

    s_spk = [1, 0]
    t_spk = [0, 1]
    loss_spk = engine.compute_spike_loss(s_spk, t_spk) # Default tau=5.0
    print(f"Spike Loss (Exp: {exp_loss}): {loss_spk}")
    if abs(loss_spk - exp_loss) > 1e-4:
        print("FAIL: Spike loss mismatch")
        sys.exit(1)

    print("=== Part 3 & 4 Tests Passed ===")

if __name__ == "__main__":
    test_parts_3_4()
