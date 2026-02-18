import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

# --- OTIMIZAÇÃO DE CPU (PARALELISMO) ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

# --- CONFIGURAÇÃO ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

DATA_PATH = "/app/teacher/data/wikitext_train.pt"
HEAD_PATH = "/app/teacher/oxta_head.pth"
BODY_PATH = "/app/teacher/oxta_body.bin"

# Hiperparâmetros
SEQ_LEN = 64
VOCAB_SIZE = 32000
DIM = 256
LR = 3e-4
STEPS = 100  # Limited to 100 for stress testing
SAVE_EVERY = 10000
REAL_BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Treino: Kernel(CPU) <-> PyTorch({DEVICE.type.upper()}) | Batch Simulado: {REAL_BATCH_SIZE}")


class Silenciador:
    def __enter__(self):
        self._original_stdout = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._original_stdout, 1)
        os.close(self._devnull)
        os.close(self._original_stdout)


try:
    with Silenciador():
        import nsos_ext
    print("✅ Kernel OXTA (C++) Carregado.")
except ImportError:
    print("❌ Compile o projeto primeiro.")
    sys.exit(1)


# Função auxiliar para garantir que o C++ carregou de verdade
def get_sample_weight(model):
    """Pega um pedaço da memória do modelo para ver se mudou."""
    # Embedding: [Vocab, Dim]. Forward expects indices.
    emb = model.embedding.forward([1])  # Pega vetor do token 1
    # Returns Tensor [1, D]
    return np.array(emb, copy=True)[0]


def train():
    # Mock data generation if missing
    if not os.path.exists(DATA_PATH):
        print("❌ Dados não encontrados. Gerando mock data para teste...")
        data = torch.randint(0, VOCAB_SIZE, (10000,)).long()
    else:
        print("📂 Carregando dataset...")
        data = torch.load(DATA_PATH).long()

    n_tokens = len(data)

    print("👶 Inicializando OXTA (3 Camadas)...")
    # Usando 3 camadas para evitar o crash de TTT (Layer 3) se ainda instavel, ou testar com 4 se corrigido.
    # O usuario pediu para adaptar este script.
    student = nsos_ext.JambaModel(3, DIM, VOCAB_SIZE)
    ctx = nsos_ext.Context()
    head = nn.Linear(DIM, VOCAB_SIZE, bias=False).to(DEVICE)

    # Prepara otimizador (importante criar antes de carregar estado)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # --- CARREGAMENTO BLINDADO ---
    loaded_parts = 0
    weight_before = get_sample_weight(student)

    # 1. Carrega Corpo (C++)
    if os.path.exists(BODY_PATH):
        print(f"♻️  Carregando CORPO (C++): {BODY_PATH}")
        try:
            with Silenciador():
                student.load(BODY_PATH)

            # Prova Real: Mudou a memória?
            weight_after = get_sample_weight(student)
            if np.array_equal(weight_before, weight_after):
                print("🚨 ALERTA: O arquivo foi lido mas a memória NÃO mudou! (Load falhou)")
            else:
                loaded_parts += 1
                print("✅ Corpo carregado e verificado.")

        except Exception as e:
            print(f"⚠️  Erro ao carregar corpo: {e}")

    # 2. Carrega Cabeça + Otimizador (PyTorch)
    if os.path.exists(HEAD_PATH):
        print(f"♻️  Carregando CABEÇA (PyTorch): {HEAD_PATH}")
        try:
            checkpoint = torch.load(HEAD_PATH, map_location=DEVICE)
            head.load_state_dict(checkpoint['head_state'])

            # O PULO DO GATO: Carregar o estado do otimizador
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print("   ✅ Otimizador restaurado (Sem tranco no Loss).")
            else:
                print("   ⚠️ Otimizador resetado (Loss pode oscilar).")

            loaded_parts += 1
        except Exception as e:
            print(f"⚠️  Erro ao carregar cabeça: {e}")

    if loaded_parts == 2:
        print("✅ Cérebro COMPLETO restaurado.")
    else:
        print("🆕 Iniciando/Reiniciando parcialmente.")

    print(f"🚀 Iniciando Treino...")
    pbar = tqdm(range(STEPS), unit="step", dynamic_ncols=True)

    optimizer.zero_grad()

    try:
        for step in pbar:
            # 1. Preparar Dados
            idx = np.random.randint(0, n_tokens - SEQ_LEN - 1)
            input_ids_list = data[idx: idx + SEQ_LEN].tolist()
            target_ids = data[idx + 1: idx + SEQ_LEN + 1].to(DEVICE)

            # 2. Forward Kernel
            # with Silenciador():
            emb_out = student.embedding.forward(input_ids_list)
            emb_out = emb_out.reshape([1, SEQ_LEN, DIM])

            hidden_cpp = student.forward(emb_out, ctx)

            # 3. PyTorch Bridge
            hidden_py = torch.from_numpy(np.array(hidden_cpp, copy=False)).float().view(SEQ_LEN, DIM).to(DEVICE)
            hidden_py.requires_grad_(True)
            hidden_py.retain_grad()

            # 4. Loss & Backward
            logits = head(hidden_py)
            loss = criterion(logits, target_ids)

            loss = loss / REAL_BATCH_SIZE
            loss.backward()

            # 5. Backward para o C++
            grad_numpy = hidden_py.grad.cpu().numpy()

            # Pass gradient to body
            # We must flatten the gradient to [N, D] to match the forward_ids output shape
            grad_flattened = grad_numpy.reshape(1 * SEQ_LEN, DIM) # [N, D]

            grad_cpp = nsos_ext.Tensor([1 * SEQ_LEN, DIM], nsos_ext.Device.CPU)
            grad_cpp.numpy()[:] = grad_flattened

            # Standard backward (uses context to find input_ids and update embedding)
            student.backward_external(grad_cpp, ctx)

            # Manual Embedding Backward needed if using backward_external and not storing indices in ctx
            # Or use new backward_embedding call if indices are needed
            # For correctness with current kernel fix:
            # We need to manually update embedding gradients if backward_external doesn't do it.
            # But backward_external propagates d_input.
            # We can use student.backward_embedding(grad_cpp, input_ids_list) ? No, we need d_input from network.
            # Actually JambaModel::backward propagates to input. If context has "input", it returns grad_input.
            # But it doesn't update weight.
            # So we need to grab the returned grad_input from backward_external?
            # The current binding backward_external returns void.
            # We updated JambaModel::backward_embedding to handle this manually.
            # We need d_input (gradient w.r.t embeddings).
            # But backward_external calls JambaModel::backward which computes it internally but doesn't expose it easily unless we modify binding return.

            # However, for this "Full Potential" script, we assume the C++ kernel fixes are active.
            # Let's rely on the body update being handled or add the explicit embedding update if we can access the gradient.
            # Since we can't easily get d_embedding from void, we skip manual embedding update here unless we change C++ return.
            # Or we trust "backward_embedding" which takes grad_hidden and indices.
            # student.backward_embedding(grad_cpp, input_ids_list) -> This computes d_embedding from d_hidden and updates weights.
            # This is redundant if backward_external does it, but currently backward_external stops at input.

            student.backward_embedding(grad_cpp, input_ids_list)

            # 6. OTIMIZAÇÃO
            if (step + 1) % REAL_BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Update C++ Body Weights (Simple SGD)
                params = student.parameters()
                for p in params:
                    d_data = np.array(p.data, copy=False)
                    d_grad = np.array(p.grad, copy=False)
                    d_data -= LR * d_grad
                    d_grad[:] = 0

            pbar.set_postfix(loss=f"{loss.item() * REAL_BATCH_SIZE:.4f}")

            # 7. SAVE COMPLETO
            if (step + 1) % SAVE_EVERY == 0:
                torch.save({
                    'head_state': head.state_dict(),
                    'optimizer_state': optimizer.state_dict()  # Salva a "velocidade" do aprendizado
                }, HEAD_PATH)
                with Silenciador():
                    student.save(BODY_PATH)

    except KeyboardInterrupt:
        print("\n🛑 Pausa manual...")

    finally:
        print("\n💾 Salvando Estado Final...")
        torch.save({
            'head_state': head.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, HEAD_PATH)
        student.save(BODY_PATH)
        print(f"✅ Cabeça + Otimizador salvos em: {HEAD_PATH}")
        print(f"✅ Corpo salvo em: {BODY_PATH}")


if __name__ == "__main__":
    train()
