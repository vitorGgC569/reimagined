
import numpy as np
import os
import sys
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "build"))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found in build/")
    sys.exit(1)

def run_induction_test():
    print("=== 🧠 Cognitive Test A: Induction Head (Copy Pattern) ===")

    vocab_size = 50
    dim = 64
    seq_len = 20
    prefix_len = 5

    model = nsos_ext.JambaModel(2, dim, vocab_size)

    # Dataset: Prefix + Pattern + ... + Prefix + Pattern
    # Simplified Induction: A B ... A -> B
    # We train on sequences where tokens repeat at fixed interval.

    steps = 500
    lr = 0.005

    print(f"Training Induction Head for {steps} steps...")

    for step in range(steps):
        # Generate Data: Random sequence of length L, repeat it twice.
        half_seq = np.random.randint(0, vocab_size, size=(seq_len // 2)).tolist()
        data = half_seq + half_seq

        input_ids = data[:-1]
        target_ids = data[1:]

        # Forward
        emb = model.embedding.forward(input_ids).reshape([1, len(input_ids), dim])
        ctx = nsos_ext.Context()
        hidden = model.forward(emb, ctx)

        # Loss
        h_np = np.array(hidden, copy=False).reshape(len(input_ids), dim)
        w_np = np.array(model.embedding.weight.data, copy=False)
        logits = np.matmul(h_np, w_np.T)

        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)

        loss = 0
        d_logits = probs.copy()

        # Check accuracy specifically on the second half (Induction)
        # Second half starts at index seq_len // 2
        induction_acc = 0
        count = 0

        for t, target in enumerate(target_ids):
            loss -= np.log(probs[t, target] + 1e-9)
            d_logits[t, target] -= 1.0

            if t >= (seq_len // 2) - 1:
                if np.argmax(probs[t]) == target:
                    induction_acc += 1
                count += 1

        loss /= len(target_ids)
        d_logits /= len(target_ids)

        # Backward
        d_hidden = np.matmul(d_logits, w_np)
        grad_t = nsos_ext.Tensor([1, len(input_ids), dim], nsos_ext.Device.CPU)
        grad_arr = np.array(grad_t, copy=False)
        grad_arr[:] = d_hidden
        model.backward_external(grad_t, ctx)

        # Update
        params = model.parameters()
        for p in params:
            d_data = np.array(p.data, copy=False)
            d_grad = np.array(p.grad, copy=False)
            d_data -= lr * d_grad
            d_grad[:] = 0

        if step % 50 == 0:
            print(f"Step {step}: Loss {loss:.4f} | Induction Acc: {induction_acc}/{count} ({(induction_acc/count)*100:.1f}%)")

def run_mqar_test():
    print("\n=== 🧠 Cognitive Test B: Multi-Query Associative Recall (MQAR) ===")
    print("(Simplified: Key-Value pairs)")

    # K: 0..9, V: 10..19. Query: K. Target: V.
    vocab_size = 30
    dim = 64
    num_pairs = 4

    model = nsos_ext.JambaModel(2, dim, vocab_size)
    lr = 0.005
    steps = 500

    print(f"Training MQAR for {steps} steps...")

    for step in range(steps):
        # Generate K-V pairs
        keys = np.random.choice(range(10), num_pairs, replace=False)
        vals = np.random.choice(range(10, 20), num_pairs, replace=False)

        # Sequence: K1 V1 K2 V2 ... Query K_random -> V_target
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([k, v])

        # Query one of the keys
        q_idx = np.random.randint(0, num_pairs)
        query_key = keys[q_idx]
        target_val = vals[q_idx]

        seq.append(query_key)
        # Target for last token is target_val

        input_ids = seq
        # We only care about the last prediction for MQAR

        emb = model.embedding.forward(input_ids).reshape([1, len(input_ids), dim])
        ctx = nsos_ext.Context()
        hidden = model.forward(emb, ctx)

        h_np = np.array(hidden, copy=False).reshape(len(input_ids), dim)
        last_h = h_np[-1] # [D]
        w_np = np.array(model.embedding.weight.data, copy=False)

        logits = np.matmul(last_h, w_np.T) # [V]

        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        loss = -np.log(probs[target_val] + 1e-9)

        d_logits = probs.copy()
        d_logits[target_val] -= 1.0

        # Backward (Need full sequence grad structure)
        # Zero grad for all tokens except last
        d_hidden_seq = np.zeros_like(h_np)
        d_hidden_seq[-1] = np.matmul(d_logits, w_np)

        grad_t = nsos_ext.Tensor([1, len(input_ids), dim], nsos_ext.Device.CPU)
        grad_arr = np.array(grad_t, copy=False)
        grad_arr[:] = d_hidden_seq

        model.backward_external(grad_t, ctx)

        # Update
        params = model.parameters()
        for p in params:
            d_data = np.array(p.data, copy=False)
            d_grad = np.array(p.grad, copy=False)
            d_data -= lr * d_grad
            d_grad[:] = 0

        if step % 50 == 0:
            pred = np.argmax(probs)
            is_correct = (pred == target_val)
            print(f"Step {step}: Loss {loss:.4f} | Pred {pred} (Target {target_val}) | {'CORRECT' if is_correct else 'FAIL'}")

if __name__ == "__main__":
    run_induction_test()
    run_mqar_test()
