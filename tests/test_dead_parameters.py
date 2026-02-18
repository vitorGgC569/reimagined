import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
    print("✅ nsos_ext imported.")
except ImportError:
    print("❌ nsos_ext not found. Compile first.")
    sys.exit(1)

# Config
SEQ_LEN = 16
VOCAB_SIZE = 100
DIM = 64
LR = 1e-3

def test_dead_parameters():
    print("🔎 Running Dead Parameter Detection CI...")

    # Init Model (3 Layers -> Mamba, FFN, TTT + MoE mixed if logic holds)
    # Layer 0: Mamba + MoE (based on jamba.cpp logic: i%2==0 -> MoE)
    # Layer 1: Mamba + FFN
    # Layer 2: TTT (if num_layers=3, i=2 is last -> TTT)
    model = nsos_ext.JambaModel(3, DIM, VOCAB_SIZE)
    ctx = nsos_ext.Context()
    head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
    criterion = nn.CrossEntropyLoss()

    # Dummy Data
    input_ids = [1, 2, 3, 4] * (SEQ_LEN // 4)
    target = torch.tensor(input_ids).long() # Dummy target

    # Forward
    hidden_cpp = model.forward_ids(input_ids, ctx)
    hidden_torch = torch.from_numpy(np.array(hidden_cpp, copy=False)).float().view(1, SEQ_LEN, DIM)
    hidden_torch.requires_grad_(True)

    # Loss
    logits = head(hidden_torch)
    loss = criterion(logits.view(-1, VOCAB_SIZE), target)
    loss.backward()

    # Backward to C++
    grad_numpy = hidden_torch.grad.numpy().reshape(1 * SEQ_LEN, DIM) # Flatten [N, D]
    grad_cpp = nsos_ext.Tensor([SEQ_LEN, DIM], nsos_ext.Device.CPU)
    grad_cpp.numpy()[:] = grad_numpy

    model.backward_external(grad_cpp, ctx)

    # Check Gradients
    params = model.parameters()
    dead_params = []

    for p in params:
        name = getattr(p, 'name', 'Unknown')
        grad = np.array(p.grad, copy=False)
        gnorm = np.linalg.norm(grad)

        print(f"   Parameter: {name:<40} | GradNorm: {gnorm:.6f}")

        if gnorm < 1e-9:
            # Exceptions? Maybe bias if unused, but we fixed b_hidden.
            # Embedding weight might have 0 grad for unused tokens, but we used input [1,2,3,4].
            # Weights for 1,2,3,4 should have grad.
            # D should have grad.
            dead_params.append(name)

    if dead_params:
        print(f"\n❌ FAILED: Found {len(dead_params)} dead parameters (Zero Gradient):")
        for name in dead_params:
            print(f"   - {name}")
        sys.exit(1)
    else:
        print("\n✅ SUCCESS: All parameters are alive and receiving gradients.")
        sys.exit(0)

if __name__ == "__main__":
    test_dead_parameters()
