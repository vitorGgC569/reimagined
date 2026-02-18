
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
    def __init__(self, in_features, out_features, rank=8): # Increased rank for harder tasks
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

# --- Trainer Helper ---
def train_task(task_name, input_seq, target_seq, model, ctx, vocab_size, d_model):
    print(f"\n>>> Treinando Tarefa: {task_name}")
    print(f"    Input: {input_seq} -> Target: {target_seq}")

    # Reset Adapter for each task to prove learnability from scratch
    adapter = TurboFusionAdapter(d_model, d_model).float()
    opt = torch.optim.Adam(adapter.parameters(), lr=0.02)

    # Input Processing
    # We treat input as a sequence. Target is the NEXT sequence or transformation.
    # To map Input -> Target sequence-to-sequence style using a Decoder-only model:
    # We typically feed [Input] and expect [Target].
    # Or feed [Input] and expect the model to continue.
    # Here we map hidden states of Input directly to Target logits (Seq2Seq behavior via Adapter).

    # Note: Input and Target lengths must match for simple mapping here,
    # or we just map last token? The prompt implies Sequence to Sequence.
    # Let's assume strict 1-to-1 token mapping for simplicity in this Proof of Concept.

    if len(input_seq) != len(target_seq):
        print("    ⚠️  Input/Target length mismatch. Truncating to minimum.")
        min_len = min(len(input_seq), len(target_seq))
        input_seq = input_seq[:min_len]
        target_seq = target_seq[:min_len]

    for epoch in range(100):
        # 1. Forward OXN (C++)
        emb = model.embedding.forward(input_seq)

        # Enable TTT (Evolution) - Model learns from the input itself
        model.session_adapt(emb, emb)

        hidden_cpp = model.forward(emb, ctx)
        h_np = np.array(hidden_cpp, copy=False)
        h_pt = torch.from_numpy(h_np).float()
        h_pt.requires_grad = True

        # 2. Forward Adapter (Python)
        h_adapted = adapter(h_pt)
        h_final = h_pt + h_adapted

        # 3. Project to Vocab
        w_emb_np = np.array(model.embedding.weight.data, copy=False)
        w_emb_pt = torch.from_numpy(w_emb_np).float()
        logits = torch.matmul(h_final, w_emb_pt.t())

        # 4. Loss
        target_pt = torch.tensor(target_seq, dtype=torch.long)
        loss = F.cross_entropy(logits, target_pt)

        # 5. Backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 6. Manual Backward to C++ (Optional, but good for completeness)
        # grad_h = h_pt.grad.numpy()
        # ... (Skipping manual C++ update to focus on Adapter learning speed)

        if epoch % 20 == 0:
            preds = torch.argmax(logits, dim=-1).tolist()
            print(f"    Epoca {epoch:02d}: Loss={loss.item():.4f} | Preds={preds}")
            if preds == target_seq:
                print(f"    ✅ Convergiu na epoca {epoch}!")
                break

    final_preds = torch.argmax(logits, dim=-1).tolist()
    print(f"    Resultado Final: {final_preds}")
    if final_preds == target_seq:
        print(f"    🏆 SUCESSO: {task_name} aprendida.")
        return True
    else:
        print(f"    ❌ FALHA: {task_name} nao convergiu.")
        return False

# --- Main Test ---
def run_tasks():
    print("=== Bateria de Ensino: Sequencia, Repeticao, Ordenacao ===")

    VOCAB = 100
    DIM = 64
    LAYERS = 2 # Deeper for logic

    model = nsos_ext.JambaModel(LAYERS, DIM, VOCAB)
    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True) # Raciocinio Ativo

    # Tarefa 1: Sequencia Numerica
    # Ensinar que depois de 1, 2, 3 vem 4, 5, 6
    # Como o modelo é "Next Token", vamos fazer: Input=[1, 2, 3], Target=[4, 5, 6]
    # Isso exige entender a relação "+3".
    t1 = train_task("Sequencia (+3)", [1, 2, 3], [4, 5, 6], model, ctx, VOCAB, DIM)

    # Tarefa 2: Repeticao
    # Input=[10, 20, 30], Target=[10, 20, 30] (Identity)
    t2 = train_task("Repeticao (Copy)", [10, 20, 30], [10, 20, 30], model, ctx, VOCAB, DIM)

    # Tarefa 3: Ordenacao
    # Input=[5, 1, 9], Target=[1, 5, 9]
    # Essa é dificil pois exige permutaçao global baseada em valor.
    t3 = train_task("Ordenacao (Sort)", [5, 1, 9], [1, 5, 9], model, ctx, VOCAB, DIM)

    print("\n=== Resumo ===")
    print(f"Sequencia: {'✅' if t1 else '❌'}")
    print(f"Repeticao: {'✅' if t2 else '❌'}")
    print(f"Ordenacao: {'✅' if t3 else '❌'}")

if __name__ == "__main__":
    run_tasks()
