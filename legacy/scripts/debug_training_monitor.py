import os
import sys
import torch
import torch.nn as nn
import numpy as np
import psutil
from tqdm import tqdm

# Setup
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
    print("✅ nsos_ext imported.")
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

# Config
SEQ_LEN = 32
VOCAB_SIZE = 1000
DIM = 128
LR = 1e-4
STEPS = 200
BATCH_SIZE = 1

def monitor_weights(model, step):
    params = model.parameters()
    print(f"\n--- Step {step} Monitor ---")
    for i, p in enumerate(params):
        data = np.array(p.data, copy=False)
        grad = np.array(p.grad, copy=False)

        norm_w = np.linalg.norm(data)
        norm_g = np.linalg.norm(grad)
        min_w, max_w = data.min(), data.max()

        name = getattr(p, 'name', f'Param_{i}')
        print(f"[{name}] NormW: {norm_w:.4f} | Range: [{min_w:.4f}, {max_w:.4f}] | NormGrad: {norm_g:.4f}")

        if np.isnan(norm_w) or np.isinf(norm_w):
            print(f"🚨 EXPLOSION DETECTED in {name}")
            return False
    return True

def run_diagnostic():
    print("🔬 Starting Deep Diagnostic Training...")

    model = nsos_ext.JambaModel(2, DIM, VOCAB_SIZE)
    ctx = nsos_ext.Context()
    head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
    criterion = nn.CrossEntropyLoss()

    # Fake Data
    data = torch.randint(0, VOCAB_SIZE, (1000,)).long()

    for step in range(STEPS):
        # 1. Data
        idx = np.random.randint(0, 1000 - SEQ_LEN - 1)
        input_ids = data[idx: idx + SEQ_LEN].tolist()
        target = data[idx+1 : idx + SEQ_LEN + 1]

        # 2. Forward
        hidden_cpp = model.forward_ids(input_ids, ctx)
        hidden_torch = torch.from_numpy(np.array(hidden_cpp, copy=False)).float().view(1, SEQ_LEN, DIM)
        hidden_torch.requires_grad_(True)

        # 3. Loss
        logits = head(hidden_torch)
        loss = criterion(logits.view(-1, VOCAB_SIZE), target.view(-1))

        print(f"Step {step}: Loss = {loss.item():.4f}")

        if np.isnan(loss.item()):
            print("🚨 Loss is NaN!")
            break

        # 4. Backward
        loss.backward()

        grad_numpy = hidden_torch.grad.numpy().reshape(1, SEQ_LEN, DIM)
        grad_cpp = nsos_ext.Tensor([1, SEQ_LEN, DIM], nsos_ext.Device.CPU)
        grad_cpp.numpy()[:] = grad_numpy

        model.backward_external(grad_cpp, ctx)

        # 5. Monitor & Update
        if not monitor_weights(model, step):
            break

        # Update (Manual SGD)
        for p in model.parameters():
            d = np.array(p.data, copy=False)
            g = np.array(p.grad, copy=False)
            # Clip gradients
            gnorm = np.linalg.norm(g)
            if gnorm > 1.0:
                g *= (1.0 / gnorm)
            d -= LR * g
            g[:] = 0 # Zero grad

    print("✅ Diagnostic Complete.")

    # Print System Stats
    process = psutil.Process()
    print(f"Memory RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    run_diagnostic()
