
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
    print(f"[Pipeline] ERROR: Could not import nsos_ext. {e}")
    sys.exit(1)

# --- TurboFusion Adapter (The Bridge) ---
class TurboFusionAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.m = nn.Parameter(torch.ones(out_features))
        self.ia3_l = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        lora_out = (x @ self.lora_A) @ self.lora_B
        dora_out = lora_out * self.m
        output = dora_out * self.ia3_l
        return output

# --- Learning Script ---
def learn_one_word():
    print("=== Desafio: Aprender uma Palavra (OLÁ -> MUNDO) ===")

    # Config
    VOCAB_SIZE = 100
    D_MODEL = 64
    LAYERS = 1

    # 1. Instanciar OXN (Aluno)
    # Vocabulario Mock:
    # 10 = "O", 11 = "L", 12 = "Á"
    # 20 = "M", 21 = "U", 22 = "N", 23 = "D", 24 = "O"

    model = nsos_ext.JambaModel(LAYERS, D_MODEL, VOCAB_SIZE)

    # Contexto com System 2 ativado (Raciocínio)
    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True)

    # Sequencia de Entrada: "OLÁ" (10, 11, 12)
    # Target: "MUNDO" (20, 21, 22, 23, 24)
    # Simulação de Next Token Prediction:
    # Input:  [10, 11, 12, 20, 21, 22, 23]
    # Target: [11, 12, 20, 21, 22, 23, 24]

    input_seq = [10, 11, 12, 20, 21, 22, 23]
    target_seq = [11, 12, 20, 21, 22, 23, 24]

    # Adaptador CART
    adapter = TurboFusionAdapter(D_MODEL, D_MODEL)
    optimizer_adapter = torch.optim.Adam(adapter.parameters(), lr=0.01)

    print(f"[Setup] Input: {input_seq}")
    print(f"[Setup] Target: {target_seq}")

    # Loop de Treino (Overfitting Intencional)
    for epoch in range(50):
        # A. Forward (C++ OXN)
        emb_tensor = model.embedding.forward(input_seq)

        # O forward do JambaModel retorna Hidden States
        hidden_states_cpp = model.forward(emb_tensor, ctx)

        # Converter para PyTorch para usar o Adapter e Loss
        h_np = np.array(hidden_states_cpp, copy=False)
        h_pt = torch.from_numpy(h_np).float()
        h_pt.requires_grad = True

        # B. Forward (Python CART)
        # O Adapter refina o pensamento do aluno
        # h_refined = h_pt + adapter(h_pt) # Residual connection
        h_adapted = adapter(h_pt)
        h_final = h_pt + h_adapted

        # C. Projeção para Vocabulario (Usando pesos do C++)
        # Pegamos os pesos de embedding do C++ para projeção (tied embeddings)
        w_emb_np = np.array(model.embedding.weight.data, copy=False)
        w_emb_pt = torch.from_numpy(w_emb_np).float()

        logits = torch.matmul(h_final, w_emb_pt.t()) # [Seq, Vocab]

        # D. Loss (Pantheon Logic - Supervised here for simplicity)
        target_pt = torch.tensor(target_seq, dtype=torch.long)
        loss = F.cross_entropy(logits, target_pt)

        # E. Backward
        optimizer_adapter.zero_grad()
        loss.backward()
        optimizer_adapter.step()

        # F. Backward para o C++ (OXN)
        # Propagar gradiente do PyTorch de volta para o Tensor C++
        grad_h_pt = h_pt.grad.numpy()

        # Criar Tensor de Gradiente C++
        grad_tensor_cpp = nsos_ext.Tensor([len(input_seq), D_MODEL], nsos_ext.Device.CPU)
        grad_buf = np.array(grad_tensor_cpp, copy=False)
        grad_buf[:] = grad_h_pt[:]

        # Executar Backward no C++
        model.backward(grad_tensor_cpp, ctx)

        # G. Update Weights C++ (SGD Manual)
        # Como o otimizador C++ nao está exposto neste teste simples, fazemos update manual
        params = model.parameters()
        for p in params:
            d = np.array(p.data, copy=False)
            g = np.array(p.grad, copy=False)
            d -= 0.05 * g # LR
            g[:] = 0 # Zero Grad

        # H. Monitoramento
        if epoch % 10 == 0 or epoch == 49:
            # Checar predição da última palavra
            last_token_logits = logits[-1]
            predicted_id = torch.argmax(last_token_logits).item()
            prob = F.softmax(last_token_logits, dim=0)[target_seq[-1]].item()
            print(f"Epoca {epoch:02d}: Loss={loss.item():.4f} | Prob('O')={prob:.4f} | Pred: {predicted_id} (Target: {target_seq[-1]})")

            # Checar se System 2 foi logado (ver output do terminal)

    print("=== Resultado Final ===")
    final_logits = logits
    decoded = [torch.argmax(l).item() for l in final_logits]
    print(f"Target : {target_seq}")
    print(f"Model  : {decoded}")

    if decoded == target_seq:
        print("✅ SUCESSO: O modelo aprendeu a sequencia perfeitamente.")
    else:
        print("⚠️ AVISO: O modelo ainda não convergiu totalmente.")

if __name__ == "__main__":
    learn_one_word()
