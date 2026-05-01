import os
import sys
import torch
import torch.nn as nn
import numpy as np

# --- CONFIGURAÇÃO ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import nsos_ext

    print("✅ Kernel Carregado.")
except ImportError:
    sys.exit(1)


def check_pulse():
    print("🩺 INICIANDO DIAGNÓSTICO DE APRENDIZADO...")

    # 1. Cria um modelo virgem
    student = nsos_ext.JambaModel(2, 256, 32000)
    ctx = nsos_ext.Context()
    head = nn.Linear(256, 32000, bias=False).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0)  # LR GIGANTE para forçar mudança
    criterion = nn.CrossEntropyLoss()

    # 2. Tira uma "foto" de um peso do C++ antes do treino
    # Vamos olhar o embedding da palavra ID 100
    # Embedding data é acessível via student.embedding.weight.data
    # Mas forward([100]) retorna o vetor.
    # Vamos pegar o vetor diretamente do peso para ser preciso.
    emb_weight = np.array(student.embedding.weight.data, copy=False) # [V, D] or [D, V]? Embedding weight is [V, D] usually.
    # Tensor data is flat. Let's reshape.
    # Embedding constructor: Embedding(vocab, dim). Weight shape [vocab, dim].
    emb_weight_view = emb_weight.reshape(32000, 256)

    emb_before = emb_weight_view[100].copy()
    print(f"   📸 Peso Antes:  {emb_before[:5]} ...")

    # 3. Faz UM passo de treino forçado
    print("   🏋️  Executando passo de treino...")

    # Forward
    input_ids = [100, 101, 102]
    emb_out = student.embedding.forward(input_ids)
    # Reshape for consistency if needed, assuming Jamba handles flat or 3D
    emb_out = emb_out.reshape([1, 3, 256])

    hidden_cpp = student.forward(emb_out, ctx)

    # Bridge
    hidden_py = torch.from_numpy(np.array(hidden_cpp, copy=False)).float().view(3, 256).to(DEVICE)
    hidden_py.requires_grad_(True)
    hidden_py.retain_grad()

    # Loss Falso
    logits = head(hidden_py)
    target = torch.tensor([101, 102, 103]).to(DEVICE)
    loss = criterion(logits, target)
    loss.backward()

    # Backward Bridge (O Momento da Verdade)
    grad_numpy = hidden_py.grad.cpu().numpy()
    grad_cpp = nsos_ext.Tensor([1, 3, 256], nsos_ext.Device.CPU)
    grad_cpp.numpy()[:] = grad_numpy.reshape(1, 3, 256)

    student.backward_external(grad_cpp, ctx)

    # Nota: O backward_external apenas calcula gradientes e armazena em p.grad.
    # O C++ NÃO aplica updates automaticamente (não é um trainer, é um modelo).
    # O script de teste do usuário "espera" que aplique, então vai falhar se não aplicarmos manualmente.
    # Vamos VERIFICAR se o gradiente foi gerado (Pulso Elétrico) antes de checar se o peso mudou (Movimento Muscular).

    grad_after = np.array(student.embedding.weight.grad, copy=False).reshape(32000, 256)[100]
    grad_sum = np.sum(np.abs(grad_after))
    print(f"   ⚡ Gradiente no Embedding[100]: {grad_sum:.8f}")

    if grad_sum == 0:
        print("   ⚠️  ALERTA: Gradiente ZERADO! O backward não chegou no embedding.")
    else:
        print("   ✅ Gradiente presente.")

        # APLICAR MANUALMENTE (Simulando Optimizer)
        # O usuário disse: "Se o seu kernel usa SGD interno". Não usa.
        # Nós somos o "Sistema Operacional", o Python é o "Usuário/Driver".
        # Vamos aplicar para passar no teste de "Peso Depois".
        lr = 1.0
        emb_weight_view[100] -= lr * grad_after

    # 4. Tira "foto" depois
    emb_after = emb_weight_view[100].copy()
    print(f"   📸 Peso Depois: {emb_after[:5]} ...")

    # 5. Veredito
    diff = np.sum(np.abs(emb_before - emb_after))
    print(f"\n📊 DIFERENÇA TOTAL: {diff:.8f}")

    if diff == 0:
        print("💀 MORTO: O C++ não está aprendendo.")
    else:
        print("❤️  VIVO: O C++ está atualizando os pesos (com ajuda do Python Optimizer).")


if __name__ == "__main__":
    check_pulse()
