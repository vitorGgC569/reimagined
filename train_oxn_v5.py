#!/usr/bin/env python3
"""
==============================================================================
NSOS/OXN TRAINING V5 - Gradiente Direto via NumPy
==============================================================================
Treina o modelo extraindo pesos, calculando gradiente em NumPy,
e atualizando os tensores C++ diretamente.

ESTRATÉGIA: Bypassa o backward C++ problemático e faz SGD manual em Python.
==============================================================================
"""

import sys
import os
import time
import random
import numpy as np

# Path do módulo compilado
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

print("=" * 70)
print("🧠 NSOS/OXN TRAINING V5 - Direct Gradient Update")
print("=" * 70)

try:
    import nsos_ext as nsos
    print("✅ nsos_ext C++ kernel carregado!")
except ImportError as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

NUM_LAYERS = 4
D_MODEL = 64
VOCAB_SIZE = 64
EPOCHS = 50
BATCHES = 10
SEQ_LEN = 8
LR = 0.01

print(f"\n   Modelo: {NUM_LAYERS}L, {D_MODEL}D, vocab={VOCAB_SIZE}")
print(f"   Treino: {EPOCHS} epochs x {BATCHES} batches, LR={LR}")

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

print("\n🏗️ Inicializando...")

model = nsos.JambaModel(NUM_LAYERS, D_MODEL, VOCAB_SIZE, nsos.Device.CPU)
params = model.parameters()
embedding = model.embedding

print(f"   ✅ {len(params)} parâmetros")

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def tensor_to_numpy(t):
    """Converte Tensor C++ para numpy array."""
    return np.array(t.numpy())

def numpy_to_tensor(arr, device=nsos.Device.CPU):
    """Cria Tensor C++ a partir de numpy array."""
    shape = list(arr.shape)
    t = nsos.Tensor(shape, device)
    np_view = np.array(t.numpy())
    np.copyto(np_view, arr.flatten().reshape(np_view.shape))
    return t

def compute_loss_and_grad(output_np, target_np):
    """Calcula MSE loss e gradiente."""
    diff = output_np - target_np
    loss = np.mean(diff ** 2)
    grad = 2 * diff / diff.size  # Gradiente normalizado
    return loss, grad

# =============================================================================
# TREINAMENTO
# =============================================================================

print("\n🚀 TREINAMENTO")
print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Δ Loss':>10} | {'Grad':>10} | Status")
print("-" * 65)

history = []
random.seed(42)
np.random.seed(42)

# Pega embedding weights iniciais
emb_weights = tensor_to_numpy(embedding.weight.data)
initial_emb_norm = np.linalg.norm(emb_weights)

for epoch in range(EPOCHS):
    epoch_losses = []
    epoch_grads = []
    
    for batch in range(BATCHES):
        # Gera tokens
        input_ids = [random.randint(1, VOCAB_SIZE-2) for _ in range(SEQ_LEN)]
        
        # Forward no modelo C++ real
        ctx = nsos.Context()
        output = model.forward_ids(input_ids, ctx)
        output_np = tensor_to_numpy(output)
        
        # Target: ruído menor que output (força modelo a aprender)
        target_np = np.random.randn(*output_np.shape).astype(np.float32) * 0.1
        
        # Loss e gradiente
        loss, grad_np = compute_loss_and_grad(output_np, target_np)
        epoch_losses.append(loss)
        epoch_grads.append(np.linalg.norm(grad_np))
        
        # Update embedding weights diretamente
        # Pega índices usados e atualiza suas linhas
        for idx in set(input_ids):
            if 0 <= idx < emb_weights.shape[0]:
                # Gradiente simplificado para embedding
                row_grad = np.mean(grad_np, axis=(0, 1)) if len(grad_np.shape) > 2 else grad_np.flatten()[:D_MODEL]
                if len(row_grad) == D_MODEL:
                    emb_weights[idx] -= LR * row_grad
        
        # Atualiza tensor C++ com novos pesos
        if batch == BATCHES - 1:  # Uma vez por epoch
            # Cria novo tensor com pesos atualizados
            new_emb = nsos.Tensor([VOCAB_SIZE, D_MODEL], nsos.Device.CPU)
            np_view = np.array(new_emb.numpy())
            np.copyto(np_view, emb_weights)
            embedding.weight.data = new_emb
    
    # Métricas
    avg_loss = np.mean(epoch_losses)
    avg_grad = np.mean(epoch_grads)
    history.append(avg_loss)
    
    if len(history) > 1:
        delta = avg_loss - history[-2]
    else:
        delta = 0
    
    # Status
    if delta < -0.001:
        status = "📉 DESCENDO"
    elif delta > 0.001:
        status = "📈 Subindo"
    else:
        status = "➡️ Estável"
    
    # Print
    if epoch < 5 or epoch >= EPOCHS - 5 or epoch % 10 == 0:
        print(f"{epoch+1:>6} | {avg_loss:>10.6f} | {delta:>+10.6f} | {avg_grad:>10.6f} | {status}")

# =============================================================================
# RESULTADOS
# =============================================================================

print("\n" + "=" * 70)
print("📊 RESULTADOS")
print("=" * 70)

# Verifica mudança nos pesos
final_emb_norm = np.linalg.norm(emb_weights)
weight_change = abs(final_emb_norm - initial_emb_norm) / initial_emb_norm * 100

# Convergência
if len(history) > 1:
    convergence = (history[0] - history[-1]) / history[0] * 100
else:
    convergence = 0

print(f"""
   Loss inicial:     {history[0]:.6f}
   Loss final:       {history[-1]:.6f}
   Convergência:     {convergence:+.2f}%
   
   Norm embedding inicial: {initial_emb_norm:.4f}
   Norm embedding final:   {final_emb_norm:.4f}
   Mudança nos pesos:      {weight_change:.2f}%
""")

# Verificações
print("   VERIFICAÇÕES:")
print(f"   [✓] Nenhum crash")
print(f"   [✓] {len(history)}/{EPOCHS} epochs")
print(f"   [{'✓' if convergence > 0 else '✗'}] Loss diminuiu")
print(f"   [{'✓' if weight_change > 0.1 else '✗'}] Pesos mudaram")

if convergence > 0 and weight_change > 0.1:
    print("\n   🎉 MODELO ESTÁ APRENDENDO!")
else:
    print("\n   ⚠️ Modelo precisa de mais ajustes")

# =============================================================================
# TESTE FINAL
# =============================================================================

print("\n" + "=" * 70)
print("🔮 INFERÊNCIA FINAL")
print("=" * 70)

test_ids = [1, 2, 3, 4, 5, 6, 7, 8]
ctx = nsos.Context()
result = model.forward_ids(test_ids, ctx)

print(f"\n   Input: {test_ids}")
print(f"   Output shape: {list(result.shape)}")
print(f"   Output norm: {result.norm():.4f}")

# Salva
try:
    model.save("oxn_trained_v5.bin")
    print("\n   💾 Modelo salvo: oxn_trained_v5.bin")
except Exception as e:
    print(f"\n   ⚠️ Erro ao salvar: {e}")

print("\n" + "=" * 70)
print("✅ TREINAMENTO COMPLETO!")
print("=" * 70)
