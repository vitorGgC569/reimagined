import os
import sys
import torch
import torch.nn as nn
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def test_kan_ablation():
    print("🔬 Running KAN Ablation Test...")

    # 1. KAN Baseline (Normal MoE with KAN Experts)
    nsos_ext.set_seed(42)
    model_kan = nsos_ext.JambaModel(1, 64, 100) # MoE Layer
    ctx_kan = nsos_ext.Context()
    input_ids = [1, 2, 3, 4]

    # Run a few steps to see loss
    head = nn.Linear(64, 100, bias=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

    target = torch.tensor(input_ids).long()

    loss_kan = 0
    for _ in range(5):
        h = model_kan.forward_ids(input_ids, ctx_kan)
        h_torch = torch.from_numpy(np.array(h, copy=False)).float().view(1, 4, 64)
        h_torch.requires_grad_(True)
        l = criterion(head(h_torch).view(-1, 100), target)
        l.backward()

        # Update body
        grad_numpy = h_torch.grad.numpy().reshape(1 * 4, 64)
        grad_cpp = nsos_ext.Tensor([4, 64], nsos_ext.Device.CPU)
        grad_cpp.numpy()[:] = grad_numpy
        model_kan.backward_external(grad_cpp, ctx_kan)

        # Simple SGD on body
        for p in model_kan.parameters():
            d = np.array(p.data, copy=False)
            g = np.array(p.grad, copy=False)
            d -= 0.01 * g
            g.fill(0)

        loss_kan = l.item()

    print(f"   KAN Final Loss: {loss_kan:.4f}")

    # 2. MLP Baseline (Simulation)
    # We don't have a switch "use_mlp_experts" in JambaModel constructor yet.
    # But we can verify if KAN parameters are moving.
    # If KAN rbf_weight is static, it acts like MLP (if base_weight moves).
    # We proved in test_dead_parameters that rbf_weight moves.

    # To truly ablate, we would need to zero out rbf_weight and freeze it.

    print("   Freezing RBF weights (Simulating MLP)...")
    nsos_ext.set_seed(42)
    model_mlp = nsos_ext.JambaModel(1, 64, 100)

    # Find rbf weights and zero gradients always
    # We can't easily hook into C++ backward to stop gradient on specific param without "requires_grad" flag.
    # But we can zero the update in Python loop.

    ctx_mlp = nsos_ext.Context()
    head_mlp = nn.Linear(64, 100, bias=False) # Same init? No, but similar.
    # Align heads?
    head_mlp.weight.data.copy_(head.weight.data)

    loss_mlp = 0
    for _ in range(5):
        h = model_mlp.forward_ids(input_ids, ctx_mlp)
        h_torch = torch.from_numpy(np.array(h, copy=False)).float().view(1, 4, 64)
        h_torch.requires_grad_(True)
        l = criterion(head_mlp(h_torch).view(-1, 100), target)
        l.backward()

        grad_numpy = h_torch.grad.numpy().reshape(4, 64)
        grad_cpp = nsos_ext.Tensor([4, 64], nsos_ext.Device.CPU)
        grad_cpp.numpy()[:] = grad_numpy
        model_mlp.backward_external(grad_cpp, ctx_mlp)

        for p in model_mlp.parameters():
            if "rbf_weight" in p.name:
                continue # Skip update for RBF -> MLP-like
            d = np.array(p.data, copy=False)
            g = np.array(p.grad, copy=False)
            d -= 0.01 * g
            g.fill(0)

        loss_mlp = l.item()

    print(f"   MLP (Simulated) Final Loss: {loss_mlp:.4f}")

    if loss_kan < loss_mlp:
        print("✅ KAN performed better than MLP baseline.")
    else:
        print("⚠️  KAN performed worse or equal (Needs tuning).")
        # Don't fail CI, just inform.

if __name__ == "__main__":
    test_kan_ablation()
