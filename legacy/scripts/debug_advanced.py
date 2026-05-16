import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import nsos_ext
except ImportError:
    print("Could not import nsos_ext.")
    sys.exit(1)

def test_ttt_gradient_direction():
    print("\n--- Test 1: TTT Gradient Direction ---")
    D = 16
    # lr = 0.1 to see effect
    ttt = nsos_ext.TTTLayer(D, D, 0.1)

    # Standard mode (hamiltonian=False)
    ttt.set_hamiltonian_mode(False)

    # Input x. Target is x (Reconstruction).
    # Loss = ||x - xW||^2
    # We want Loss to decrease.

    x = nsos_ext.Tensor([1, 1, D], nsos_ext.Device.CPU)
    x_np = np.ones((1, 1, D), dtype=np.float32)
    x_view = np.array(x.numpy(), copy=False)
    x_view[:] = x_np

    # Initial forward
    out1 = ttt.forward(x) # Updates W internally
    out1_np = np.array(out1.numpy())
    loss1 = np.mean((x_np - out1_np)**2)
    print(f"Initial Loss (approx): {loss1}")

    # Second forward (W has changed)
    # If W updated correctly (descent), reconstruction of SAME input should be better?
    # TTT updates W to map x -> x.
    # So next time we see x, error should be lower.

    # Reset input (forward modifies x? No, x is const ref)
    out2 = ttt.forward(x)
    out2_np = np.array(out2.numpy())
    loss2 = np.mean((x_np - out2_np)**2)
    print(f"Loss after 1 step: {loss2}")

    if loss2 > loss1:
        print("FAIL: Loss INCREASED. Likely Gradient Ascent.")
    else:
        print("PASS: Loss Decreased.")

def test_mamba_stability():
    print("\n--- Test 2: Mamba Stability (Exp Decay) ---")
    # Need to trigger large negative dt.
    # Mamba2SSD inputs: u [B, L, D]
    # dt is derived from projection of u.
    # We can't easily control dt directly from python unless we mock internals.
    # But we can feed huge inputs?

    model = nsos_ext.JambaModel(1, 64, 128)
    x = nsos_ext.Tensor([1, 10, 64], nsos_ext.Device.CPU)
    x_np = np.random.normal(0, 100.0, (1, 10, 64)).astype(np.float32) # Huge values
    x_view = np.array(x.numpy(), copy=False)
    x_view[:] = x_np

    try:
        out = model.forward_embedding(x) # Bypass embedding lookup, feed raw tensor as 'hidden'
        # Wait, forward_embedding expects x to be output of embedding layer (hidden state)
        # JambaModel::forward_embedding takes Tensor x.

        # We need to call Mamba inside.
        # JambaModel::forward(x) calls forward_embedding.

        # Let's just call model.forward(x) treating x as embeddings.
        # But wait, JambaModel::forward checks if input is int or float?
        # The binding `forward` takes Tensor x.
        # Inside C++, it passes to `forward_embedding`.
        # Correct.

        ctx = nsos_ext.Context()
        # In bindings, forward(x, ctx) is correct.
        # But wait, bindings use py::arg with default nullptr.
        # However, pybind sometimes is picky.
        # forward(x, ctx)

        out = model.forward(x, ctx)
        norm = out.norm()
        print(f"Output Norm: {norm}")
        if np.isnan(norm) or np.isinf(norm):
            print("FAIL: Mamba produced NaN/Inf")
        else:
            print("PASS: Mamba survived huge input")

    except Exception as e:
        print(f"Crash: {e}")

def test_system2_performance():
    print("\n--- Test 3: System 2 Performance (Graph) ---")
    model = nsos_ext.JambaModel(1, 64, 128)

    # Create Context with force_system2
    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True)

    # Sequence length 512
    S = 512
    x = nsos_ext.Tensor([1, S, 64], nsos_ext.Device.CPU)
    x_np = np.random.normal(0, 1.0, (1, S, 64)).astype(np.float32)
    x_view = np.array(x.numpy(), copy=False)
    x_view[:] = x_np

    start = time.time()
    out = model.forward(x, ctx)
    end = time.time()
    print(f"Time for S={S}: {end - start:.4f}s")

    # If O(S^2), 1024 should be 4x slower.
    # We can just verify it runs without crashing for now.

if __name__ == "__main__":
    test_ttt_gradient_direction()
    test_mamba_stability()
    test_system2_performance()
