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

def test_moe_forcing():
    print("🔬 Running MoE Expert Forcing Test...")

    # Model config
    # Layer 0 is MoE (i%2==0) if it's not the last layer (TTT)
    # So we need at least 2 layers.
    model = nsos_ext.JambaModel(2, 64, 100)
    ctx = nsos_ext.Context()
    head = nn.Linear(64, 100, bias=False)
    criterion = nn.CrossEntropyLoss()

    input_ids = [1, 2, 3, 4]
    target = torch.tensor(input_ids).long()

    # Identify MoE Layer
    # We access via parameters() finding names
    params = model.parameters()
    gate_param = None

    for p in params:
        # print(f"Found param: {p.name}")
        if "layers.0.moe.gate.weight" in p.name:
            gate_param = p
            break

    if not gate_param:
        print("❌ Could not find Router Gate weights. Available:")
        for p in params:
            print(f"   - {p.name}")
        sys.exit(1)

    print("✅ Found Router Gate.")

    # Assuming Gate Weight is [Dim, NumExperts]
    # We need to confirm shape.
    # Gate tensor: shape vector.
    gate_shape = gate_param.data.shape
    num_experts = gate_shape[1]
    dim = gate_shape[0]

    print(f"   Gate Shape: {gate_shape} (Dim={dim}, Experts={num_experts})")

    # Loop over each expert to force it
    all_alive = True

    for k in range(num_experts):
        print(f"\n🧪 Forcing activation of Expert {k}...")

        # 1. Hack Weights: Set column k to +100, others to -100
        # Access raw data buffer
        # This assumes memory layout [Dim, Experts] (Row major?) or [Dim, Experts].
        # In Tensor.cpp: flat index = i*strides.
        # Tensor is usually row-major. [D, E].
        # index(d, e) = d * E + e.

        # We need to write to the C++ tensor memory.
        # Use numpy view if available or manual copy?
        # The binding exposes .data as buffer.

        # Reset gradients
        for p in params:
            np.array(p.grad, copy=False).fill(0)

        # Manipulate Gate Weight (Use gentle values to avoid NaN explosion in gradients)
        w_np = np.array(gate_param.data, copy=False)
        w_np.fill(-2.0) # Suppress all
        w_np[:, k] = 2.0 # Promote k

        # 2. Forward
        # Clear context to avoid state contamination
        ctx = nsos_ext.Context()
        hidden = model.forward_ids(input_ids, ctx)

        # 3. Backward
        h_torch = torch.from_numpy(np.array(hidden, copy=False)).float().view(1, 4, 64)
        h_torch.requires_grad_(True)
        loss = criterion(head(h_torch).view(-1, 100), target)
        loss.backward()

        grad_numpy = h_torch.grad.numpy().reshape(1 * 4, 64)
        grad_cpp = nsos_ext.Tensor([4, 64], nsos_ext.Device.CPU)
        grad_cpp.numpy()[:] = grad_numpy

        model.backward_external(grad_cpp, ctx)

        # 4. Check Gradient of Expert k
        # Find expert param
        expert_grad_norm = 0.0
        found_expert = False

        expert_prefix = f"layers.0.moe.expert.{k}."
        for p in params:
            if expert_prefix in p.name:
                g = np.array(p.grad, copy=False)
                gn = np.linalg.norm(g)
                expert_grad_norm += gn
                found_expert = True
                # print(f"   - {p.name}: {gn}")

        if not found_expert:
            print(f"   ❌ Expert {k} parameters not found!")
            all_alive = False
            continue

        print(f"   Expert {k} Total Grad Norm: {expert_grad_norm:.6f}")

        if expert_grad_norm > 1e-6:
            print(f"   ✅ Expert {k} is ALIVE.")
        else:
            print(f"   ❌ Expert {k} is DEAD (Zero Gradient) despite forcing.")
            all_alive = False

    if all_alive:
        print("\n✅ SUCCESS: All experts can be activated and trained.")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Some experts are dead.")
        sys.exit(1)

if __name__ == "__main__":
    test_moe_forcing()
