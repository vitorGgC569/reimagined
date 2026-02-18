
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 0. Environment Setup ---
sys.path.append(os.path.abspath("OXN/nsos/build"))
sys.path.append(os.path.abspath("build"))

try:
    import nsos_ext
except ImportError as e:
    print(f"[ERROR] Could not import nsos_ext. {e}")
    sys.exit(1)

# --- TurboFusion Adapter ---
class TurboFusionAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.m = nn.Parameter(torch.ones(out_features))
        self.ia3_l = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        lora_out = (x @ self.lora_A) @ self.lora_B
        dora_out = lora_out * self.m
        output = dora_out * self.ia3_l
        return output

# --- OOD Sorting Test ---
def test_ood_sorting():
    print("=== Teste de Generalização OOD (Sorting N=3 -> N=6) ===")

    VOCAB = 10 # Digits 0-9
    DIM = 64
    LAYERS = 2

    model = nsos_ext.JambaModel(LAYERS, DIM, VOCAB)

    # --- Semantic Initialization (Crucial for Sorting) ---
    print("[Setup] Inicializando Embeddings Semânticos (0..9)...")
    embed_weights = np.array(model.embedding.weight.data, copy=False)
    embed_weights = embed_weights.reshape((VOCAB, DIM))

    base_vec = np.random.randn(DIM).astype(np.float32)
    direction = np.random.randn(DIM).astype(np.float32)
    direction /= np.linalg.norm(direction)

    # 0 maps to base, 9 maps to base + 9*dir
    # Geometrically ordered line in high-dim space
    for i in range(10):
        embed_weights[i] = base_vec + (i * 1.0 * direction)

    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True) # Graph Active

    adapter = TurboFusionAdapter(DIM, DIM)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.01)

    # --- Training Data (Length 3) ---
    # Generate 50 random samples of len 3
    train_samples = []
    for _ in range(50):
        seq = np.random.permutation(10)[:3].tolist()
        tgt = sorted(seq)
        train_samples.append((seq, tgt))

    # --- Testing Data (Length 6) ---
    # Generate 10 samples of len 6
    test_samples = []
    for _ in range(10):
        seq = np.random.permutation(10)[:6].tolist()
        tgt = sorted(seq)
        test_samples.append((seq, tgt))

    print(f"[Treino] Iniciando em {len(train_samples)} listas de tamanho 3...")

    # --- Training Loop ---
    for epoch in range(100):
        total_loss = 0
        np.random.shuffle(train_samples)

        for inp_seq, tgt_seq in train_samples:
            # Forward
            emb = model.embedding.forward(inp_seq)
            model.session_adapt(emb, emb) # TTT active

            hidden = model.forward(emb, ctx)
            h_pt = torch.from_numpy(np.array(hidden, copy=False)).float()
            h_pt.requires_grad = True

            h_final = h_pt + adapter(h_pt)

            # Project using tied embeddings
            w_emb_np = np.array(model.embedding.weight.data, copy=False)
            w_emb_pt = torch.from_numpy(w_emb_np).float()
            logits = torch.matmul(h_final, w_emb_pt.t())

            # Loss
            target = torch.tensor(tgt_seq, dtype=torch.long)
            loss = F.cross_entropy(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Avg Loss = {total_loss/len(train_samples):.4f}")
            if total_loss/len(train_samples) < 0.01:
                print("    ✅ Convergiu no Treino (N=3)!")
                break

    # --- Testing Loop (OOD) ---
    print("\n[Teste] Avaliando em listas de tamanho 6 (Nunca vistas)...")
    correct_lists = 0

    for inp_seq, tgt_seq in test_samples:
        # Inference
        emb = model.embedding.forward(inp_seq)
        model.session_adapt(emb, emb) # TTT still active to adapt to NEW context length

        # System 2 v2.0: REASONING LOOP (Recurrence)
        # Instead of 1 pass, we run a cognitive loop using the WDD Scratchpad
        # This gives the model "time" to sort longer sequences.
        steps = 4
        hidden = model.run_reasoning_loop(emb, steps)

        h_pt = torch.from_numpy(np.array(hidden, copy=False)).float()

        h_final = h_pt + adapter(h_pt)

        w_emb_np = np.array(model.embedding.weight.data, copy=False)
        w_emb_pt = torch.from_numpy(w_emb_np).float()
        logits = torch.matmul(h_final, w_emb_pt.t())

        preds = torch.argmax(logits, dim=-1).tolist()

        status = "✅" if preds == tgt_seq else "❌"
        print(f"    In: {inp_seq} -> Pred: {preds} (Tgt: {tgt_seq}) {status}")
        if preds == tgt_seq: correct_lists += 1

    print("\n=== Resultado OOD ===")
    acc = correct_lists / len(test_samples)
    print(f"Acurácia N=6: {acc*100:.1f}%")

    if acc > 0.5:
        print("🏆 SUCESSO: O modelo generalizou a regra de ordenação para sequencias longas!")
    else:
        print("⚠️ FALHA: O modelo não generalizou o comprimento (Expected Behavior for vanilla Transformers).")

if __name__ == "__main__":
    test_ood_sorting()
