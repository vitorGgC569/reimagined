import torch
import time
import sys

# Try to import the compiled extension
try:
    # Need to make sure we can import the built module.
    # Usually setup.py install puts it in site-packages, but we might be running locally.
    sys.path.append('.')
    import peft_torch
except ImportError as e:
    print(f"Failed to import peft_torch: {e}")
    sys.exit(1)

def verify():
    print("--- Verifying PyTorch Integration ---")

    in_dim = 128
    out_dim = 128
    rank = 8

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Instantiate Model
    model = peft_torch.TurboFusion(in_dim, out_dim, rank).to(device)
    print("Model instantiated successfully.")

    # 1. Forward Pass Test
    x = torch.randn(32, in_dim).to(device)
    try:
        y = model(x)
        print(f"Forward pass successful. Output shape: {y.shape}")
        assert y.shape == (32, out_dim)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        sys.exit(1)

    # 2. Backward Pass Test (Autograd)
    try:
        target = torch.randn(32, out_dim).to(device)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        print("Backward pass successful. Gradients computed.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        sys.exit(1)

    # 3. Optimization Test
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.step()
    optimizer.zero_grad()
    print("Optimizer step successful.")

    # 4. Benchmarking (Quick)
    print("\n--- Quick Benchmark (Inference) ---")
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(x)
    end = time.time()
    avg_time = (end - start) / num_iters * 1000
    print(f"Average Inference Time: {avg_time:.4f} ms per batch (Batch Size 32)")

if __name__ == "__main__":
    verify()
