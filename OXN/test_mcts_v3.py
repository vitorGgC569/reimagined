import sys
import os
import numpy as np

# Verificar se nsos_ext está presente no diretório atual
if os.path.exists("nsos_ext.pyd"):
    print("nsos_ext.pyd encontrado no diretório atual.")
else:
    print("AVISO: nsos_ext.pyd não encontrado no diretório atual.")

try:
    import nsos_ext
    print("nsos_ext carregado com sucesso!")
    print(f"Localização do nsos_ext: {nsos_ext.__file__}")
except ImportError as e:
    print(f"Erro ao carregar nsos_ext: {e}")
    sys.exit(1)

def test_mcts_reasoning():
    print("\n[Teste] Iniciando Verificação do MCTSReasoning (System-2)...")
    
    # 1. Configurar Estado Inicial
    root_state = nsos_ext.Tensor([3], nsos_ext.Device.CPU, 0.0)
    root_state_np = root_state.numpy()
    root_state_np[0] = 1.0
    root_state_np[1] = 0.0
    root_state_np[2] = -1.0
    
    # 2. Configuração do MCTS
    config = nsos_ext.MCTSConfig()
    config.num_simulations = 100
    config.max_depth = 5
    config.num_children_per_expansion = 3
    
    # 3. Função de Avaliação (Mock)
    def mock_evaluator(state):
        return float(np.sum(state.numpy()))

    # 4. Inicializar MCTS
    mcts = nsos_ext.MCTSReasoning(root_state, mock_evaluator, config)
    
    print(f"Buscando com {config.num_simulations} simulações...")
    mcts.search()
    
    # 5. Verificar Resultados
    nodes = mcts.num_nodes()
    root_visits = mcts.root_visits()
    best_value = mcts.get_best_value()
    best_state = mcts.get_best_state()
    
    print(f"Busca finalizada.")
    print(f" - Nós ativos no Pool: {nodes}")
    print(f" - Visitas na raiz: {root_visits}")
    print(f" - Melhor valor encontrado: {best_value:.4f}")
    if best_state and best_state.size > 0:
        print(f" - Melhor estado (primeiros 3): {best_state.numpy()[:3]}")
    
    # Verificar se o valor melhorou
    initial_value = mock_evaluator(root_state)
    print(f" - Valor Inicial: {initial_value:.4f}")
    
    if best_value > initial_value:
        print("[SUCESSO] MCTS encontrou um estado superior via raciocínio.")
    else:
        print("[ALERTA] MCTS não encontrou melhoria.")

    # 6. Verificar Caminho de Raciocínio
    path = mcts.get_best_path()
    print(f"Caminho de raciocínio (profundidade {len(path)-1}):")
    for i, s in enumerate(path):
        val = mock_evaluator(s)
        print(f"  [{i}] State Sum: {val:.4f}")

if __name__ == "__main__":
    test_mcts_reasoning()
