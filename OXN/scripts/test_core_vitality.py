import sys
import os
import numpy as np

# Add build dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import nsos_ext
except ImportError:
    print("Erro: nsos_ext não encontrado.")
    sys.exit(1)

print("==========================================")
print("   NSOS NANO-TEST V8 (Direct Bypass)      ")
print("   Target: MEMORY SANITIZATION & LIFE     ")
print("==========================================")

d_model = 16
vocab_size = 128
tokenizer = nsos_ext.Tokenizer()

# 1. Apenas a Camada de Saída (BitFastKAN)
# Isolamos o JambaModel pois ele pode ter memória suja interna
head = nsos_ext.BitFastKANLayer(d_model, vocab_size)

# 2. Pesos Mestres (Controlados via Python)
master_weights = nsos_ext.Tensor([d_model, vocab_size], nsos_ext.Device.CPU, 0.0)
master_view = np.asarray(master_weights)
# Inicialização Limpa e Segura
master_view[:] = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.1

# Ponte para memória C++ da Head
print(f"[Debug] head.base_weight.data shape: {head.base_weight.data.shape}")
head_view = np.asarray(head.base_weight.data)
print(f"[Debug] head_view shape: {head_view.shape}")

print("[System] JambaModel disconnected. Wiring Input -> Head direct.")

for epoch in range(5):
    print(f"\n--- Epoch {epoch + 1} ---")

    # A. Entrada Higienizada
    # Criamos o tensor e GARANTIMOS que ele não tem lixo de memória
    seq_len = 10
    input_tensor = nsos_ext.Tensor([seq_len, d_model], nsos_ext.Device.CPU, 0.0)
    input_view = np.asarray(input_tensor)
    input_view.fill(0.0)  # Limpeza explícita
    # Adiciona sinal forte
    input_view[:] = np.random.randn(seq_len, d_model).astype(np.float32)

    # B. Forward Simplificado (Pula Jamba/JEPA)
    # Testa puramente a matemática da BitNet e a tokenização
    logits = head.forward(input_tensor)

    # C. Otimização Manual (Update dos pesos)
    # Gradiente Simulado
    grad_noise = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.2

    # Update: W = W - grad
    master_view -= grad_noise

    # CLAMP (Trava de Segurança)
    np.clip(master_view, -1.0, 1.0, out=master_view)

    # Quantização e Upload para C++
    head_view[:] = np.round(master_view).T

    # D. Diagnóstico
    logits_np = np.asarray(logits)
    out_ids = np.argmax(logits_np, axis=1)

    # Check de Vida
    if np.isnan(logits_np).any():
        print("   [FATAL] A própria camada BitFastKAN está gerando NaNs.")
    else:
        # Mostra os valores para vermos a variação
        print(f"   Logits Sample: {logits_np[0][:4]}")
        out_text = tokenizer.decode(out_ids.tolist())
        print(f"   Output: '{out_text}'")

print("\n==========================================")
print("Se o texto mudou (ex: 'abac...'), o sistema de tokens e pesos funciona.")
