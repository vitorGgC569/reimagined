
import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found in build/")
    sys.exit(1)

def run_convergence_test():
    print("=== 📉 Convergence Benchmark (Associative Recall) ===")
    print("Config: LR=1e-3, Warmup=20, Steps=200, Batch=4 (Accum)")

    # Task: Copy Task / Associative Recall
    vocab_size = 100
    dim = 64
    seq_len = 16

    # Hyperparams Tuning
    base_lr = 0.001 # Reduced from 0.01
    max_steps = 200 # Increased from 100
    warmup_steps = 20
    batch_size = 4 # Gradient Accumulation

    model = nsos_ext.JambaModel(2, dim, vocab_size)

    losses = []

    # Optimization State
    accum_steps = 0

    for step in range(max_steps):
        # Scheduler (Linear Warmup)
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            # Cosine Decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        # Data Gen
        data = np.random.randint(0, vocab_size, size=(seq_len)).tolist()
        input_ids = data[:-1]
        target_ids = data[1:]

        # Forward
        emb = model.embedding.forward(input_ids).reshape([1, len(input_ids), dim])
        ctx = nsos_ext.Context()
        hidden = model.forward(emb, ctx)

        # Numpy Bridge
        h_np = np.array(hidden, copy=False).reshape(len(input_ids), dim)
        w_np = np.array(model.embedding.weight.data, copy=False)

        logits = np.matmul(h_np, w_np.T)

        # Softmax
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)

        loss = 0
        d_logits = probs.copy()

        correct = 0
        for t, target in enumerate(target_ids):
            loss -= np.log(probs[t, target] + 1e-9)
            d_logits[t, target] -= 1.0
            if np.argmax(probs[t]) == target:
                correct += 1

        loss /= len(target_ids)
        d_logits /= len(target_ids)
        losses.append(loss)

        # Backward
        d_hidden = np.matmul(d_logits, w_np)

        grad_t = nsos_ext.Tensor([1, len(input_ids), dim], nsos_ext.Device.CPU)
        grad_arr = np.array(grad_t, copy=False)
        grad_arr[:] = d_hidden

        # Backward accumulates gradients in C++ parameters
        model.backward_external(grad_t, ctx)

        accum_steps += 1

        # Update Step (Batched)
        if accum_steps >= batch_size:
            params = model.parameters()
            for p in params:
                d_data = np.array(p.data, copy=False)
                d_grad = np.array(p.grad, copy=False)

                # Apply Update
                # Note: Gradients are sum of batch_size steps.
                # We want average? Or just LR scaling.
                # Standard Accum: w = w - lr * (sum_grad / batch)
                d_data -= lr * (d_grad / batch_size)

                # Zero Grad
                d_grad[:] = 0
            accum_steps = 0

        if step % 20 == 0:
            print(f"Step {step}: Loss {loss:.4f} | Acc {correct}/{len(target_ids)} | LR {lr:.5f}")

    print(f"Final Loss: {losses[-1]:.4f}")

    # ASCII Plot
    print("\n[Loss Curve]")
    max_l = max(losses)
    min_l = min(losses)
    height = 10
    if max_l == min_l: max_l += 1e-6

    for i in range(height):
        thresh = max_l - (i * (max_l - min_l) / height)
        line = ""
        for l in losses:
            if l >= thresh: line += "*"
            else: line += " "
        print(f"{thresh:.2f} | {line}")

if __name__ == "__main__":
    run_convergence_test()
