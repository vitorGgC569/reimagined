import os
import sys

# Ensure the build directory is in the path
build_path = os.path.abspath(os.path.join(os.getcwd(), "build", "Release"))
if build_path not in sys.path:
    sys.path.append(build_path)

try:
    import nsos_ext
    print("[SUCCESS] nsos_ext loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Failed to load nsos_ext: {e}")
    print(f"Looked in: {build_path}")
    sys.exit(1)

def test_ttt_adaptation():
    print("\n--- Testing TTT Sequence Adaptation ---")
    dim = 64
    hidden = 128
    lr = 0.01
    
    layer = nsos_ext.TTTLayer(dim, hidden, lr)
    layer.set_hamiltonian_mode(True)
    layer.set_temperature(0.05)
    layer.set_friction(0.8)
    
    # Create a batch of sequences
    # Shape: [Batch=2, Seq=10, Dim=64]
    x = nsos_ext.Tensor([2, 10, dim])
    x.fill(1.0) # Uniform input
    
    print("Running initial forward pass...")
    y1 = layer.forward(x)
    print(f"Output shape: {y1.shape()}")
    
    # Get values at t=0 and t=9 for the first batch
    # We expect them to be different because the layer adapts
    # However, since input is constant, if it were a static layer, outputs would be identical.
    
    data = y1.to_list()
    # y1 is [2, 10, 128]
    # Token 0: batch 0, seq 0 -> offset 0 to 127
    # Token 9: batch 0, seq 9 -> offset 9*128 to 10*128-1
    
    t0_vals = data[0:10]
    t9_vals = data[9*hidden : 9*hidden + 10]
    
    diff = sum(abs(a - b) for a, b in zip(t0_vals, t9_vals))
    print(f"Adaptation diff (t0 vs t9): {diff}")
    
    if diff > 0:
        print("[PASS] Layer adapted over the sequence.")
    else:
        print("[WARNING] No adaptation detected. Check if LR is too low or logic is static.")

def test_ttt_backward():
    print("\n--- Testing TTT Backward Pass (Meta-Learning) ---")
    dim = 32
    hidden = 64
    layer = nsos_ext.TTTLayer(dim, hidden, 0.001)
    
    x = nsos_ext.Tensor([1, 5, dim])
    x.fill(0.5)
    
    y = layer.forward(x)
    
    # Grad output
    grad_y = nsos_ext.Tensor(y.shape())
    grad_y.fill(0.1)
    
    print("Running backward pass...")
    grad_x = layer.backward(grad_y)
    print(f"Grad input shape: {grad_x.shape()}")
    
    # Check if params have gradients
    params = layer.parameters()
    has_grad = False
    for p in params:
        # Check norm of gradients
        g = p.grad
        if g:
            # Simple sum check
            g_list = g.to_list()
            s = sum(abs(v) for v in g_list)
            if s > 0:
                print(f"Param '{p.name}' received gradients (sum={s:.6f})")
                has_grad = True
    
    if has_grad:
        print("[PASS] Gradients propagated to parameters.")
    else:
        print("[FAIL] No gradients detected in parameters.")

if __name__ == "__main__":
    test_ttt_adaptation()
    test_ttt_backward()
    print("\n[VERIFICATION COMPLETE]")
