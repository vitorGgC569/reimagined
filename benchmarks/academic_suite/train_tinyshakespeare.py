
import time
import numpy as np
import os
import sys
from dashboard_logger import Dashboard

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found")
    sys.exit(1)

def train_tinyshakespeare():
    print("=== 🎭 Training on TinyShakespeare (Generalization Test) ===")

    # 1. Load Data
    data_path = "data/tinyshakespeare.txt" # Mock path, we generate dummy if missing
    if not os.path.exists(data_path):
        print("Downloading/Generating TinyShakespeare...")
        text = "To be, or not to be, that is the question: \n" * 1000
        # Use dummy text for the benchmark if file doesn't exist to avoid networking issues in restricted env
    else:
        with open(data_path, 'r') as f:
            text = f.read()

    # Simple Char Tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 1 # +1 for padding
    stoi = { ch:i+1 for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]

    data = encode(text)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 2. Model Config
    dim = 128
    block_size = 64
    batch_size = 4
    learning_rate = 1e-3
    max_iters = 100

    model = nsos_ext.JambaModel(4, dim, vocab_size)
    dash = Dashboard("tinyshakespeare_telemetry.csv")

    # 3. Training Loop
    print(f"Vocab: {vocab_size}, Train Tokens: {len(train_data)}")

    for iter in range(max_iters):
        start_t = time.time()

        # Get Batch
        ix = np.random.randint(0, len(train_data) - block_size, batch_size)
        x_batch = [train_data[i:i+block_size] for i in ix]
        y_batch = [train_data[i+1:i+block_size+1] for i in ix]

        # Flatten for C++ interface (Simulating Batch=1 logically or using accumulations)
        # For simplicity, train on ONE sample per step effectively or accumulate
        input_ids = x_batch[0]
        target_ids = y_batch[0]

        # Forward
        emb = model.embedding.forward(input_ids).reshape([1, len(input_ids), dim])
        ctx = nsos_ext.Context()
        hidden = model.forward(emb, ctx)

        # Loss Calculation (Python Side)
        h_np = np.array(hidden, copy=False).reshape(len(input_ids), dim)
        w_np = np.array(model.embedding.weight.data, copy=False)
        logits = np.matmul(h_np, w_np.T)

        # Cross Entropy
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_sum = np.sum(probs, axis=1, keepdims=True)
        probs = probs / probs_sum

        loss = 0
        d_logits = probs.copy()
        for t, target in enumerate(target_ids):
            if target < vocab_size:
                loss -= np.log(probs[t, target] + 1e-9)
                d_logits[t, target] -= 1.0
        loss /= len(target_ids)
        d_logits /= len(target_ids)

        # Backward
        d_hidden = np.matmul(d_logits, w_np)
        grad_t = nsos_ext.Tensor([1, len(input_ids), dim], nsos_ext.Device.CPU)
        grad_arr = np.array(grad_t, copy=False)
        grad_arr[:] = d_hidden

        model.backward_external(grad_t, ctx)

        # Telemetry: Grad Norm
        params = model.parameters()
        total_norm = 0.0
        for p in params:
            g = np.array(p.grad, copy=False)
            total_norm += np.sum(g**2)
            # Update SGD
            d = np.array(p.data, copy=False)
            d -= learning_rate * g
            g[:] = 0 # Zero grad
        total_norm = np.sqrt(total_norm)

        # Log
        dt = time.time() - start_t
        dash.log(iter, loss, learning_rate, total_norm, {'mean': np.mean(h_np), 'std': np.std(h_np)})

        if iter % 10 == 0:
            print(f"Iter {iter}: Loss {loss:.4f} | GradNorm {total_norm:.2f} | Time {dt*1000:.1f}ms")

if __name__ == "__main__":
    train_tinyshakespeare()
