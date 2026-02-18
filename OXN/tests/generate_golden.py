
import torch
import os
import sys

# Try to import nsos_ext
try:
    import nsos_ext
except ImportError:
    # Build path hack
    sys.path.append('/app/build')
    sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
    import nsos_ext

def generate_golden():
    print("📀 Generating Golden Record...")
    os.makedirs("tests/data", exist_ok=True)

    # 1. Determinism
    nsos_ext.set_seed(42)
    torch.manual_seed(42)

    # 2. Create Input
    B, S, D = 1, 8, 64
    x = torch.randn(B, S, D, dtype=torch.float32)

    # 3. Run Model
    # Important: Use CPU for baseline to avoid GPU indeterminism (atomicAdd order)
    model = nsos_ext.JambaModel(2, D, 100, nsos_ext.Device.CPU)

    t_in = nsos_ext.Tensor(list(x.shape), nsos_ext.Device.CPU)
    t_in.copy_from(nsos_ext.Tensor.from_blob(x.data_ptr(), list(x.shape), nsos_ext.Device.CPU))

    out_nsos = model.forward(t_in)
    out_torch = torch.from_numpy(out_nsos.numpy())

    # 4. Save
    torch.save(x, "tests/data/golden_input.pt")
    torch.save(out_torch, "tests/data/golden_output_v1.pt")

    print(f"✅ Saved Golden Record (Shape: {out_torch.shape})")
    print(f"   Input Hash: {x.sum().item():.4f}")
    print(f"   Output Hash: {out_torch.sum().item():.4f}")

if __name__ == "__main__":
    generate_golden()
