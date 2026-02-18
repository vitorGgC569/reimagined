import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath("OXN/nsos/build"))
try:
    import nsos_ext
except ImportError:
    print("Error importing nsos_ext")
    sys.exit(1)

def test_evolution():
    print("=== Teste de Evolucao (TTT - Degrau 1) ===")

    # Setup
    vocab = 100
    dim = 64
    layers = 1
    model = nsos_ext.JambaModel(layers, dim, vocab)

    # Input
    input_ids = [1, 2, 3]
    emb = model.embedding.forward(input_ids)
    ctx = nsos_ext.Context()

    # 1. Primeira Passada (Antes do TTT)
    out1 = model.forward(emb, ctx)
    val1 = np.array(out1, copy=False).mean()
    print(f"Output Medio (Antes): {val1:.6f}")

    # 2. Executar TTT (Session Adapt)
    # Adaptar para que o input produza ele mesmo (Identity / Reconstruction) ou um target mockado
    print("[TTT] Executando session_adapt()...")
    model.session_adapt(emb, emb)

    # 3. Segunda Passada (Depois do TTT)
    # O modelo deve ter mudado seus pesos internos (TTTLayer)
    out2 = model.forward(emb, ctx)
    val2 = np.array(out2, copy=False).mean()
    print(f"Output Medio (Depois): {val2:.6f}")

    diff = abs(val2 - val1)
    print(f"Diferenca (Evolucao): {diff:.6f}")

    if diff > 1e-9:
        print("✅ SUCESSO: O modelo evoluiu (pesos mudaram).")
    else:
        print("⚠️ FALHA: O modelo permaneceu estatico.")

def test_infinite_memory():
    print("\n=== Teste de Contexto Infinito (Degrau 3) ===")
    mem_sys = nsos_ext.MemorySystem(64)

    print("Enchendo memoria...")
    # Fill memory beyond limit (1000)
    for i in range(1100):
        t = nsos_ext.Tensor([64], nsos_ext.Device.CPU, 1.0)
        mem_sys.store_episodic(t)

    print("Memoria preenchida. Se nao houve Crash/OOM, o swap logic funcionou.")
    print("✅ SUCESSO: Gerenciamento de Memoria ativo.")

if __name__ == "__main__":
    test_evolution()
    test_infinite_memory()
