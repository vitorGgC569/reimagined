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

# --- SISTEMA DE ARQUIVOS DUPLO (CÉREBRO COMPLETO) ---
HEAD_PATH = "/app/teacher/oxta_head.pth"  # PyTorch (Classificador)
BODY_PATH = "/app/teacher/oxta_body.bin"  # C++ (Memória Profunda)

# Hiperparâmetros
SEQ_LEN = 64
VOCAB_SIZE = 32000
DIM = 256
LR = 3e-4
STEPS = 50000
SAVE_EVERY = 1000

# Gradient Accumulation
REAL_BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Treino: Kernel(CPU) <-> PyTorch({DEVICE.type.upper()}) | Batch Simulado: {REAL_BATCH_SIZE}")
print(f"🚀 Threads de CPU Ativas: {torch.get_num_threads()}")


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


def train():
    # Mock Data se não existir
    if not os.path.exists(DATA_PATH):
        print("⚠️  Dados não encontrados. Gerando mock data...")
        data = torch.randint(0, VOCAB_SIZE, (10000,)).long()
    else:
        print("📂 Carregando dataset...")
        data = torch.load(DATA_PATH).long()

    n_tokens = len(data)

    print("👶 Inicializando OXTA (4 Camadas - TTT Ativo)...")
    # Configurado para 4 camadas conforme pedido (0,1,2 Mamba, 3 TTT)
    student = nsos_ext.JambaModel(4, DIM, VOCAB_SIZE)
    ctx = nsos_ext.Context()
    head = nn.Linear(DIM, VOCAB_SIZE, bias=False).to(DEVICE)

    # --- CARREGAMENTO DO CÉREBRO (Corpo + Cabeça) ---
    loaded_parts = 0

    # 1. Carrega Corpo (C++)
    if os.path.exists(BODY_PATH):
        print(f"♻️  Carregando CORPO (C++): {BODY_PATH}")
        try:
            with Silenciador():
                student.load(BODY_PATH)
            loaded_parts += 1
        except Exception as e:
            print(f"⚠️  Erro ao carregar corpo: {e}")

    # 2. Carrega Cabeça (PyTorch)
    if os.path.exists(HEAD_PATH):
        print(f"♻️  Carregando CABEÇA (PyTorch): {HEAD_PATH}")
        checkpoint = torch.load(HEAD_PATH, map_location=DEVICE)
        head.load_state_dict(checkpoint['head_state'])
        loaded_parts += 1

    if loaded_parts == 0:
        print("🆕 Iniciando treino do zero.")
    elif loaded_parts == 1:
        print("⚠️  Aviso: Carregamento parcial (Loss pode oscilar no início).")
    else:
        print("✅ Cérebro completo restaurado.")

    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

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
            # with Silenciador(): # Debug: Comentei para ver logs se crashar
            emb_out = student.embedding.forward(input_ids_list)
            # Ensure reshape for 3D input expectations in some layers
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
            grad_cpp = nsos_ext.Tensor([1, SEQ_LEN, DIM], nsos_ext.Device.CPU)
            grad_cpp.numpy()[:] = grad_numpy.reshape(1, SEQ_LEN, DIM)

            # with Silenciador():
            student.backward_external(grad_cpp, ctx)

            # 6. OTIMIZAÇÃO (Gradient Accumulation)
            if (step + 1) % REAL_BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Update C++ parameters (SGD simple stub for body)
                # In real scenario, we should extract params and use PyTorch optimizer or C++ optimizer.
                # For this script, we assume C++ params are static or updated internally?
                # Wait, JambaModel doesn't have internal optimizer step called here.
                # We need to update C++ weights!
                # Adding simple SGD update for demonstration stability
                params = student.parameters()
                for p in params:
                    d_data = np.array(p.data, copy=False)
                    d_grad = np.array(p.grad, copy=False)
                    d_data -= LR * d_grad
                    d_grad[:] = 0

            pbar.set_postfix(loss=f"{loss.item() * REAL_BATCH_SIZE:.4f}")

            # 7. SAVE DUPLO (A cada 1000 passos)
            if (step + 1) % SAVE_EVERY == 0:
                torch.save({'head_state': head.state_dict()}, HEAD_PATH)
                with Silenciador():
                    student.save(BODY_PATH)

    except KeyboardInterrupt:
        print("\n🛑 Pausa manual...")

    finally:
        print("\n💾 Salvando Estado Final...")
        torch.save({'head_state': head.state_dict()}, HEAD_PATH)
        student.save(BODY_PATH)
        print(f"✅ Cabeça salva em: {HEAD_PATH}")
        print(f"✅ Corpo salvo em: {BODY_PATH}")


if __name__ == "__main__":
    train()
