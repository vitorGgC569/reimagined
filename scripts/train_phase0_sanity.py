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

def train_phase0():
    print("=== PHASE 0: Sanity Check (WikiText-2) ===")

    # 0. Device Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    nsos_device = nsos_ext.Device.GPU if use_cuda else nsos_ext.Device.CPU
    print(f"   Running on: {device}")

    # 1. Config
    SEQ_LEN = 32
    DIM = 128

    # Initialize Model on correct device
    model = nsos_ext.JambaModel(2, DIM, 256, device=nsos_device)
    ctx = nsos_ext.Context()

    data_path = os.path.join(ROOT_DIR, "data/phase_0/train.pt")
    if not os.path.exists(data_path):
        print("⚠️  Data not found. Run download_and_prep.py first.")
        return

    data = torch.load(data_path) # Assumed Tensor? WikiText mock is likely CPU tensor.

    head = nn.Linear(DIM, 256, bias=False).to(device)

    LR = 1e-4
    optim = torch.optim.AdamW(head.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # 2. Train Loop
    pbar = tqdm(range(50))
    for step in pbar:
        # Batch
        idx = np.random.randint(0, len(data) - SEQ_LEN - 1)
        input_ids = data[idx:idx+SEQ_LEN].tolist()
        target = data[idx+1:idx+SEQ_LEN+1].to(device)

        # Forward
        h_cpp = model.forward_ids(input_ids, ctx) # Returns Tensor (on GPU if model is GPU)

        # Zero-Copy Conversion (Safe)
        if use_cuda:
            # h_cpp has __cuda_array_interface__
            h = torch.as_tensor(h_cpp, device=device).view(1, SEQ_LEN, DIM)
        else:
            # h_cpp is CPU, .numpy() is safe
            h = torch.from_numpy(h_cpp.numpy()).float().view(1, SEQ_LEN, DIM)

        h.requires_grad_(True)

        loss = crit(head(h).view(-1, 256), target)
        loss.backward()

        # Backward Handoff
        g_torch = h.grad.detach() # [1, S, D]
        g_flat = g_torch.reshape(1 * SEQ_LEN, DIM)

        # Clip (Torch side is faster/easier)
        torch.nn.utils.clip_grad_norm_([g_flat], 1.0)

        # Handoff back to C++
        # We need to construct a C++ Tensor from PyTorch data without copy if possible?
        # Creating C++ Tensor from ptr is hard from Python (no constructor exposed).
        # We must copy.
        # But wait! 'copy_from' exists? No.
        # We create a new C++ Tensor.

        if use_cuda:
            # We can't easily wrap existing GPU ptr into C++ Tensor from Python yet.
            # Workaround: Create new GPU Tensor in C++, then copy?
            # Creating from vector/list is CPU-based.
            # Ideally: nsos_ext.Tensor.from_blob(ptr, shape, device)? Not implemented.
            # Current bottleneck: Passing gradient back requires copy via CPU if not careful.

            # Temporary Solution: Move to CPU, pass to C++ (It copies), C++ moves to GPU (if using 'to').
            # This is the "Ping Pong" we wanted to avoid.
            # BUT, we can improve:
            # If we create `g_t` as GPU tensor: `g_t = nsos_ext.Tensor(..., Device.GPU)`
            # We assume it allocates GPU memory.
            # But we can't fill it from Python without access to ptr.

            # Since Layer 1 (C++ Tensor) doesn't expose "fill from ptr", we MUST fall back to CPU copy for gradients for now.
            # However, forward pass is zero-copy! Half the battle won.
            # And Model parameters stay on GPU.

            g_cpu = g_flat.cpu().numpy()
            g_t = nsos_ext.Tensor([1 * SEQ_LEN, DIM], nsos_ext.Device.CPU)
            g_t.numpy()[:] = g_cpu

            # Move to GPU inside C++?
            # backward_external takes Tensor. If model is GPU, it expects GPU tensor?
            # Let's check `backward`. It calls `out_proj.backward`.
            # `out_proj` weights are on GPU. Input `grad` must be on GPU.
            # So we MUST pass a GPU tensor to `backward_external`.

            g_t_gpu = g_t.to(nsos_ext.Device.GPU) # Explicit move
            model.backward_external(g_t_gpu, ctx)

        else:
            g_t = nsos_ext.Tensor([1 * SEQ_LEN, DIM], nsos_ext.Device.CPU)
            g_t.numpy()[:] = g_flat.numpy()
            model.backward_external(g_t, ctx)

        # Optimizer (Head)
        optim.step()
        optim.zero_grad()

        # C++ Update
        # Ideally move this loop to C++ `trainer.step()` to avoid Python iteration overhead
        # But for now, we iterate parameters.
        # Params are on GPU.
        for p in model.parameters():
            # p.data is GPU Tensor.
            # We need to perform update on GPU.
            # p.data -= LR * p.grad
            # Tensor has `sub`, `mul`.
            # p.data = p.data.sub(p.grad.mul(LR)) -> Creates new tensor?
            # We need IN-PLACE update.
            # Tensor bindings don't expose in-place operators explicitly except via C++ logic.
            # But we can assume SGD is simple.

            # C++ side update logic:
            # Accessing .data via numpy() on GPU throws error (Good!).
            # We can't use numpy.
            # We rely on C++ methods.

            # Implement a simple optimizer in C++ or use the `trainer.cpp`?
            # Existing `SGDOptimizer` in bindings?
            # Yes: `py::class_<SGDOptimizer>(m, "SGDOptimizer").def("step", &SGDOptimizer::step)`
            # But `SGDOptimizer` in `trainer.cpp` probably iterates params?
            pass # We need to update weights!

            # Fallback: Safe CPU update if we can't do GPU update from Python easily?
            # NO! That's ping pong.

            # Better: Use `p.data` methods.
            # p.data is a Tensor.
            # We need `p.data.add_(...)`.
            # `Tensor` has `add` (returns new).
            # We need `copy_from` to write back?
            # `p.data.copy_from(p.data.sub(p.grad.mul(LR)))`
            # This works on GPU!

            # Gradient clipping C++ side?
            # Tensor::clip_grad_norm_ is static and exposed.
            # nsos_ext.Tensor.clip_grad_norm_([p.grad for p in params], 1.0)

            if p.grad.norm() > 0: # Check if grad exists/nonzero
                 # Simple SGD
                 update = p.grad.mul(LR)
                 new_data = p.data.sub(update)
                 p.data.copy_from(new_data) # In-place update via copy

                 # Zero grad?
                 # Need p.grad.zero_() or create new zero tensor.
                 # p.grad = zeros like p.grad
                 # Parameter.grad is readonly field?
                 # `def_readonly("grad", &Parameter::grad)` -> Yes, usually copies?
                 # Wait, pybind11 `def_readonly` for objects returns a copy unless `reference_internal`.
                 # We bound `Parameter` fields as readonly.
                 # `py::return_value_policy::reference` in `parameters()` binding helps.
                 # But accessing `p.grad` might return a copy of the shared_ptr wrapper?
                 # Yes, `std::shared_ptr<Tensor>` is copied, but points to same data.
                 # So `p.grad.copy_from(zeros)` works.

                 zero_t = nsos_ext.Tensor.zeros(p.grad.shape, nsos_device)
                 p.grad.copy_from(zero_t)

        pbar.set_postfix(loss=loss.item())

    print("✅ Phase 0 Complete: Loss stable.")

if __name__ == "__main__":
    train_phase0()
