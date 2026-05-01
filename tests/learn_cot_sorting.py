
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.generate_reasoning_data import generate_dataset

# --- Setup ---
sys.path.append(os.path.abspath("OXN/nsos/build"))
sys.path.append(os.path.abspath("build"))
try:
    import nsos_ext
except ImportError:
    print("Error importing nsos_ext")
    sys.exit(1)

# --- Adapter ---
class TurboFusionAdapter(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, vocab) # Project directly to vocab for text generation
        )
    def forward(self, x):
        return self.net(x)

# --- Tokenizer Mock ---
# We need to map text tokens (CMP, SWAP, numbers) to IDs.
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<think>": 1, "</think>": 2, "CMP": 3, "SWAP.": 4, "KEEP.": 5, "STATE": 6, "END.": 7, ".": 8, "[": 9, "]": 10, ",": 11}
        # Add numbers
        for i in range(10): self.vocab[str(i)] = 12 + i
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.size = len(self.vocab)

    def encode(self, text):
        # Naive split by space, handling punctuation if attached
        tokens = []
        # Cleanup string rep of list
        text = text.replace("[", " [ ").replace("]", " ] ").replace(",", " , ")
        parts = text.split()
        for p in parts:
            if p in self.vocab: tokens.append(self.vocab[p])
            else: tokens.append(0) # unk
        return tokens

    def decode(self, ids):
        return " ".join([self.rev_vocab.get(i, "?") for i in ids])

# --- Trainer ---
def train_cot():
    print("=== Treinamento Supervisionado de Chain of Thought (CoT) ===")

    tokenizer = SimpleTokenizer()
    VOCAB = tokenizer.size + 10 # Buffer
    DIM = 64
    LAYERS = 2

    model = nsos_ext.JambaModel(LAYERS, DIM, VOCAB)
    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True)

    # Adapter now acts as the Language Head
    head = TurboFusionAdapter(DIM, VOCAB)
    opt = torch.optim.Adam(head.parameters(), lr=0.005)

    # Generate Training Data (Length 3)
    # We train on SHORT lists to learn the algorithm
    raw_data = generate_dataset(num_samples=50, length=3)

    print(">>> Iniciando Treino (Ensinando o Algoritmo Bubble Sort)...")

    for epoch in range(50):
        total_loss = 0

        for sample in raw_data:
            # Prepare Input: "Input: [x, y, z]" -> Target: "<think>..."
            # For simplicity, we feed input, and train to predict Target autoregressively.
            # But JambaModel expects fixed tensor input?
            # We used 'embedding.forward' before.

            # Full Sequence: Input + Target
            full_text = sample['input'] + " " + sample['target']
            input_ids = tokenizer.encode(full_text)

            # Teacher Forcing:
            # Input to model: ids[:-1]
            # Target: ids[1:]

            inp_tensor_seq = input_ids[:-1]
            target_tensor_seq = input_ids[1:]

            # Forward OXN
            emb = model.embedding.forward(inp_tensor_seq)
            # TTT Active (Learning to Learn)
            model.session_adapt(emb, emb)

            hidden = model.forward(emb, ctx) # [Seq, Dim]
            h_pt = torch.from_numpy(np.array(hidden, copy=False)).float()

            # Project to Vocab
            logits = head(h_pt) # [Seq, Vocab]

            # Loss
            tgt = torch.tensor(target_tensor_seq, dtype=torch.long)
            loss = F.cross_entropy(logits, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Avg Loss = {total_loss/len(raw_data):.4f}")

    print("✅ Treino Concluído.")

    # --- Test Generalization (N=6) ---
    print("\n>>> Teste de Generalização (N=6) com CoT...")
    test_sample = generate_dataset(1, 6)[0]
    input_text = test_sample['input']
    print(f"    Input: {input_text}")

    # Generation Loop
    # Feed input -> Predict <think> -> Loop until </think> -> Predict Result
    current_ids = tokenizer.encode(input_text)

    print("    Gerando Raciocínio (Brain Monitor):")
    generated = []

    for _ in range(50): # Max steps limit
        emb = model.embedding.forward(current_ids)
        # REASONING LOOP ACTIVE
        # The model uses the scratchpad to stabilize this long thought process
        hidden = model.run_reasoning_loop(emb, 2)

        h_pt = torch.from_numpy(np.array(hidden, copy=False)).float()
        last_h = h_pt[-1] # Predict next token

        logit = head(last_h)
        next_id = torch.argmax(logit).item()

        word = tokenizer.decode([next_id])
        generated.append(word)
        current_ids.append(next_id)

        if word == "END." or word == "]": # Stop condition
            break

    full_gen = " ".join(generated)
    print(f"    Saída do Modelo: {full_gen}")

    if "SWAP" in full_gen or "CMP" in full_gen:
        print("🏆 SUCESSO: O modelo tentou aplicar o algoritmo (gerou passos de raciocínio)!")
    else:
        print("⚠️ FALHA: O modelo não gerou raciocínio estruturado.")

if __name__ == "__main__":
    train_cot()
