import sys
import os
import torch
import torch.nn as nn

# Adiciona o diretório 'python' ao path
sys.path.append(os.path.join(os.getcwd(), 'python'))
import oxta_mem

def run_circuit_ai():
    print("--- IA CAUSAL: CONSULTANDO BANCO DE VERDADES FÍSICAS ---")
    
    # 1. Carregar os dados brutos e reconstruir a lógica
    if not os.path.exists("nn/circuit_memory.pt"):
        print("Erro: Memória do circuito não encontrada. Rode 'simulation/circuit_solver.py' primeiro.")
        return
        
    checkpoint = torch.load("nn/circuit_memory.pt", weights_only=False)
    heads = checkpoint["heads"]
    store = checkpoint["store"]
    
    print("Memória carregada e reconstruída. Simulando recall causal...")
    
    # 2. IA de Recall (Viagem no Tempo)
    # Vamos pedir para a IA nos dizer o que aconteceu 2 segundos atrás (20 steps)
    # O Oxta-Mem faz isso nativamente navegando no Merkle-DAG
    
    # Pegamos o estado mais recente (Agora)
    now_addr = heads.get("rc_node_0")
    
    print(f"\nEstado Atual (Agora):")
    current_state = store[now_addr]["val"]
    print(f"  t={current_state[0]:.1f}s | Vin={current_state[1]:.1f}V | Vout={current_state[2]:.3f}V")
    
    # TESTE DE RECALL (VIAGEM NO TEMPO T-20)
    print("\nNavegando no Grafo Causal para t - 2.0s...")
    
    # Caminhamos no grafo buscando o nó 20 steps atrás
    ptr = now_addr
    for _ in range(20):
        ptr = store[ptr]["prev"]
        
    past_state = store[ptr]["val"]
    
    print(f"Resultado da Consulta Geodésica:")
    print(f"  t={past_state[0]:.1f}s | Vin={past_state[1]:.1f}V | Vout={past_state[2]:.3f}V")
    
    # 3. Validação de Lógica Causal
    # Se t era 10.0s e voltamos 20 steps (de 0.1s), t deve ser 8.0s.
    if abs(past_state[0] - 8.0) < 0.01:
        print("\n✅ SUCESSO: A IA recuperou a verdade física exata sem cálculos!")
        print("Diferente de uma predição aproximada, a Oxta-Mem fornece o dado histórico íntegro.")

if __name__ == "__main__":
    run_circuit_ai()
