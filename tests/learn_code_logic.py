
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
    def __init__(self, in_features, out_features, rank=16): # Higher rank for logic
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

# --- Logic Test ---
def test_conditional_logic():
    print("=== Teste de Lógica Condicional (IF/ELSE) ===")

    # Vocabulario de Código (Mock)
    # 0-9: Numeros literais
    # 10: x, 11: =, 12: if, 13: >, 14: :, 15: y, 16: else, 17: print
    vocab_map = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        'x': 10, '=': 11, 'if': 12, '>': 13, ':': 14, 'y': 15, 'else': 16, 'print': 17
    }

    # Snippet Template:
    # x = {VAL}
    # if x > 3:
    #    y = 1
    # else:
    #    y = 0
    # print y -> {RESULT}

    # Base sequence structure (tokens)
    # [x, =, VAL, if, x, >, 3, :, y, =, 1, else, :, y, =, 0, print, y]
    base_seq = [10, 11, -1, 12, 10, 13, 3, 14, 15, 11, 1, 16, 14, 15, 11, 0, 17, 15]

    def get_sample(val):
        seq = base_seq.copy()
        seq[2] = val # Inject x value
        target = 1 if val > 3 else 0
        return seq, [target]

    # Training Data
    # Learn on x=2 (False -> 0) and x=5 (True -> 1)
    # Test on x=4 (True -> 1) and x=1 (False -> 0)
    train_samples = [get_sample(2), get_sample(5), get_sample(0), get_sample(6)]
    test_samples = [get_sample(4), get_sample(1)]

    VOCAB = 20
    DIM = 64
    LAYERS = 2

    model = nsos_ext.JambaModel(LAYERS, DIM, VOCAB)

    # NOTE: We removed the manual Semantic Initialization hack.
    # We now rely on the C++ 'RIERASS Anchor Injection' which happens automatically
    # inside Embedding::forward for tokens 0-9 when force_system2 is active (or by default in this prototype).

    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True)

    adapter = TurboFusionAdapter(DIM, DIM)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.01)

    print(">>> Treinando Lógica (x > 3)...")

    for epoch in range(100):
        total_loss = 0
        correct = 0

        # Shuffle train
        np.random.shuffle(train_samples)

        for inp_seq, tgt_seq in train_samples:
            # 1. Forward
            emb = model.embedding.forward(inp_seq)
            # TTT Active
            model.session_adapt(emb, emb)

            hidden_cpp = model.forward(emb, ctx)
            h_pt = torch.from_numpy(np.array(hidden_cpp, copy=False)).float()
            h_pt.requires_grad = True

            # Adapter
            h_final = h_pt + adapter(h_pt)

            # Project
            w_emb_np = np.array(model.embedding.weight.data, copy=False)
            w_emb_pt = torch.from_numpy(w_emb_np).float()
            logits = torch.matmul(h_final, w_emb_pt.t()) # [Seq, Vocab]

            # Loss on LAST token (the result)
            last_logit = logits[-1].unsqueeze(0)
            target = torch.tensor(tgt_seq, dtype=torch.long)

            loss = F.cross_entropy(last_logit, target)

            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pred = torch.argmax(last_logit).item()
            if pred == tgt_seq[0]: correct += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss:.4f} | Acc={correct}/{len(train_samples)}")
            if correct == len(train_samples) and total_loss < 0.01:
                print("    ✅ Convergiu no Treino!")
                break

    print("\n>>> Validando em Dados Não Vistos (Generalização de Regra)")
    results = []
    for val, (inp_seq, tgt_seq) in zip([4, 1], test_samples):
        # Inference
        emb = model.embedding.forward(inp_seq)
        # Note: We keep TTT enabled during inference to allow adaptation to the specific prompt context
        model.session_adapt(emb, emb)

        hidden_cpp = model.forward(emb, ctx)
        h_pt = torch.from_numpy(np.array(hidden_cpp, copy=False)).float()

        h_final = h_pt + adapter(h_pt)

        w_emb_np = np.array(model.embedding.weight.data, copy=False)
        w_emb_pt = torch.from_numpy(w_emb_np).float()
        logits = torch.matmul(h_final, w_emb_pt.t())

        pred = torch.argmax(logits[-1]).item()

        status = "✅" if pred == tgt_seq[0] else "❌"
        print(f"    Input: x = {val} | Esperado: {tgt_seq[0]} | Modelo: {pred} {status}")
        results.append(pred == tgt_seq[0])

    if all(results):
        print("🏆 SUCESSO: O modelo aprendeu a lógica condicional 'x > 3'.")
    else:
        print("⚠️ PARCIAL: O modelo decorou o treino mas falhou na generalização total.")

if __name__ == "__main__":
    test_conditional_logic()
