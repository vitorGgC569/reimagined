import sys
import os
import time
import numpy as np

# Adiciona o diretório de build ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import nsos_ext
except ImportError:
    print("Erro: Compile o projeto primeiro (nsos_ext não encontrado).")
    sys.exit(1)

print("==========================================")
print("   NSOS NANO-TEST V2 (Symmetry Breaker)   ")
print("==========================================")

# 1. Configuração
d_model = 16
vocab_size = 128
model = nsos_ext.JambaModel(4, d_model)
head = nsos_ext.BitFastKANLayer(d_model, vocab_size)
tokenizer = nsos_ext.Tokenizer()

# --- CORREÇÃO 1: Dimensões Corretas ---
# O Otimizador deve casar com a camada que ele treina (Head: d_model x vocab_size)
optimizer = nsos_ext.MuonOptimizer([d_model, vocab_size], 0.05)

# --- CORREÇÃO 2: Pesos Mestres (FP32) ---
# Em BitNet, mantemos uma cópia de alta precisão para acumular gradientes
master_weights = nsos_ext.Tensor([d_model, vocab_size], nsos_ext.Device.CPU, 0.0)

print("[Data] Generating Synthetic Noise via Numpy Bridge...")

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1} ---")

    # Geramos uma entrada aleatória (tokens simulados)
    seq_len = 10

    # Cria o Tensor no C++
    input_tensor = nsos_ext.Tensor([seq_len, d_model], nsos_ext.Device.CPU, 0.0)

    # --- A MÁGICA (Injeção de Ruído) ---
    # Graças ao fix nos bindings, acessamos a memória do C++ via Numpy sem copiar
    input_view = np.array(input_tensor, copy=False)
    # Preenchemos com ruído real para quebrar a simetria dos neurônios
    input_view[:] = np.random.randn(seq_len, d_model).astype(np.float32)

    # Pipeline: Coconut -> JEPA -> BitFastKAN
    latent = model.forward_thought(input_tensor, 1)
    emb = model.forward_embedding(latent)
    logits = head.forward(emb)

    # Otimização Muon
    # Usamos um gradiente simulado que casa com o tamanho dos pesos (d_model * vocab_size)
    # input_tensor é [seq, d], mas pesos são [d, vocab]. Precisamos de um gradiente compatível.
    # Criamos um gradiente aleatório do tamanho certo.
    grad_tensor = nsos_ext.Tensor([d_model, vocab_size], nsos_ext.Device.CPU, 0.0)
    grad_view = np.array(grad_tensor, copy=False)
    grad_view[:] = np.random.randn(d_model, vocab_size).astype(np.float32)

    # Atualizamos o master_weights e quantizamos o resultado na head.base_weight
    optimizer.step_and_quantize(head.base_weight, grad_tensor, master_weights)

    # Decodificação
    logits_np = np.array(logits, copy=False)
    out_ids = np.argmax(logits_np, axis=1)

    # O tokenizer C++ simples mapeia 0->'a', 1->'b', etc.
    # Com o ruído, os IDs devem variar, gerando texto "alienígena" mas variado
    out_text = tokenizer.decode(out_ids.tolist())
    print(f"Output: '{out_text}'")

print("\n==========================================")
print("Teste Finalizado. Se o texto variou, a engine está viva.")
