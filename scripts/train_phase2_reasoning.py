import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    print("❌ nsos_ext not found.")
    sys.exit(1)

def train_phase2():
    print("=== PHASE 2: Reasoning (GSM8K) ===")

    # 0. Device Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    nsos_device = nsos_ext.Device.GPU if use_cuda else nsos_ext.Device.CPU
    print(f"   Running on: {device}")

    # 1. Config: MoE Active
    SEQ_LEN = 64
    DIM = 256
    model = nsos_ext.JambaModel(2, DIM, 256, device=nsos_device)
    ctx = nsos_ext.Context()

    data_path = os.path.join(ROOT_DIR, "data/phase_2/train.pt")
    if not os.path.exists(data_path):
        print("⚠️  Data not found.")
        return

    data = torch.load(data_path)
    head = nn.Linear(DIM, 256, bias=False).to(device)

    LR = 1e-4
    optim = torch.optim.AdamW(head.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # 2. Train Loop
    pbar = tqdm(range(20))
    for step in pbar:
        idx = np.random.randint(0, len(data) - SEQ_LEN - 1)
        input_ids = data[idx:idx+SEQ_LEN].tolist()
        target = data[idx+1:idx+SEQ_LEN+1].to(device)

        h_cpp = model.forward_ids(input_ids, ctx)

        if use_cuda:
            h = torch.as_tensor(h_cpp, device=device).view(1, SEQ_LEN, DIM)
        else:
            h = torch.from_numpy(h_cpp.numpy()).float().view(1, SEQ_LEN, DIM)

        h.requires_grad_(True)

        loss = crit(head(h).view(-1, 256), target)
        loss.backward()

        g_torch = h.grad.detach().reshape(SEQ_LEN, DIM)
        torch.nn.utils.clip_grad_norm_([g_torch], 1.0)

        if use_cuda:
            g_cpu = g_torch.cpu().numpy()
            g_t = nsos_ext.Tensor([SEQ_LEN, DIM], nsos_ext.Device.CPU)
            g_t.numpy()[:] = g_cpu
            model.backward_external(g_t.to(nsos_ext.Device.GPU), ctx)
        else:
            g_t = nsos_ext.Tensor([SEQ_LEN, DIM], nsos_ext.Device.CPU)
            g_t.numpy()[:] = g_torch.numpy()
            model.backward_external(g_t, ctx)

        optim.step()
        optim.zero_grad()

        # C++ Update
        for p in model.parameters():
            if p.grad.norm() > 0:
                update = p.grad.mul(LR)
                new_data = p.data.sub(update)
                p.data.copy_from(new_data)
                zero_t = nsos_ext.Tensor.zeros(p.grad.shape, nsos_device)
                p.grad.copy_from(zero_t)

        pbar.set_postfix(loss=loss.item())

    print("✅ Phase 2 Complete: Reasoning tasks processed.")

if __name__ == "__main__":
    train_phase2()
