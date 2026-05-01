import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==============================================================================
# 1. O KERNEL DE MEMÓRIA GEODÉSICA (A Estrutura 4D)
# ==============================================================================
class GeodesicMemoryCore:
    def __init__(self):
        # Heads: Dicionário que aponta para o "Agora" de cada variável
        # Em C++, isso seriam ponteiros de memória real.
        self.heads = {}

        # Store: O "HD" ou "RAM" física onde os dados vivem
        self.store = {}

    def write(self, token_id, vector_value):
        """
        Cria um novo nó no tempo.
        Não sobrescreve o anterior, apenas atualiza o ponteiro 'head'.
        """
        # Recupera o ponteiro para o passado (t-1)
        prev_ptr = self.heads.get(token_id, None)

        # Cria endereço único (hash simulado)
        new_node_addr = f"{token_id}_{len(self.store)}"

        node_data = {
            "val": vector_value.clone().detach() if torch.is_tensor(vector_value) else vector_value, 
            "prev": prev_ptr,                     # Link Causal para o passado
            "timestamp": len(self.store)
        }

        self.store[new_node_addr] = node_data
        self.heads[token_id] = new_node_addr # Atualiza a referência temporal
        return new_node_addr

    def read_latest(self, token_id):
        """
        Consulta O(1).
        A rede pede 'X', o kernel entrega o conteúdo do endereço atual de 'X'.
        """
        addr = self.heads.get(token_id)
        if addr:
            return self.store[addr]["val"]
        else:
            return None

# ==============================================================================
# 2. A REDE NEURAL AUMENTADA (O Cérebro)
# ==============================================================================
class CausalAugmentedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CausalAugmentedNet, self).__init__()

        self.memory = GeodesicMemoryCore()

        # Codificador de Entrada (Sentidos)
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Camada de Fusão (Onde 'Ver' encontra 'Lembrar')
        # Entrada dobrada: recebe o embedding da entrada + o embedding da memória
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        # Decodificador de Saída (Ação)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, token_id, input_vector, is_training_store=False):
        """
        token_id: Identificador da variável (ex: "Var_X")
        input_vector: O dado visual (pode ser Zeros se for um teste de memória)
        is_training_store: Se True, grava o input atual na memória 4D.
        """

        # 1. Processa o que está vendo agora
        embedding_now = self.relu(self.input_layer(input_vector))

        # 2. Se for etapa de escrita, grava na memória ANTES de consultar
        if is_training_store:
            # Gravamos o embedding bruto ou o processado?
            # Para este teste, gravamos o embedding processado.
            self.memory.write(token_id, embedding_now)

        # 3. Consulta a Memória Geodésica (Time Travel)
        memory_val = self.memory.read_latest(token_id)

        if memory_val is None:
            # Se não tem memória, usa zeros (amnésia)
            memory_val = torch.zeros_like(embedding_now)

        # 4. Fusão Neural
        # Concatena o "Agora" com o "Passado"
        combined = torch.cat((embedding_now, memory_val), dim=0)

        # A rede decide quanto peso dar para a memória vs entrada
        fused_state = self.relu(self.fusion_layer(combined))

        output = self.output_layer(fused_state)
        return output

# ==============================================================================
# 3. O PROTOCOLO DE EXPERIMENTO (Validação Científica)
# ==============================================================================

def run_experiment():
    print("--- INICIANDO PROTOCOLO DE VALIDAÇÃO: NEURAL-SYMBOLIC 4D ---")

    # Hiperparâmetros
    INPUT_SIZE = 4
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 4
    LEARNING_RATE = 0.02
    EPOCHS = 200 # Aumentado para garantir convergência na tarefa difícil

    # Inicialização
    model = CausalAugmentedNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Dados de Teste (One-Hot Encoded simples)
    # X = [1, 0, 0, 0]
    # Y = [0, 1, 0, 0]
    val_X = torch.tensor([1.0, 0.0, 0.0, 0.0])
    val_Y = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # O "Input Cego" (Usado para forçar a rede a usar a memória)
    blind_input = torch.zeros(4)

    print(f"\n[FASE 1] Treinamento com 'Dropout Sensorial' ({EPOCHS} épocas)...")
    print("Objetivo: Ensinar a rede a ignorar input vazio e confiar no ponteiro de memória.")

    history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss_total = 0

        # --- PASSO A: ESCRITA (Observação) ---
        # A rede vê o dado real e o grava na memória.
        # Não calculamos loss aqui pois é apenas "absorção de dados".
        with torch.no_grad(): # Economiza gradiente, pois é só storage
            model("Var_X", val_X, is_training_store=True)
            model("Var_Y", val_Y, is_training_store=True)

        # --- PASSO B: LEITURA (Recall Forçado) ---
        # A rede recebe ZEROS na entrada.
        # Ela precisa responder X ou Y baseada apenas no token_id.

        # Pergunta: "Quem é X?" (Input: 0000) -> Resposta Esperada: val_X
        pred_X = model("Var_X", blind_input, is_training_store=False)
        loss_X = criterion(pred_X, val_X)

        # Pergunta: "Quem é Y?" (Input: 0000) -> Resposta Esperada: val_Y
        pred_Y = model("Var_Y", blind_input, is_training_store=False)
        loss_Y = criterion(pred_Y, val_Y)

        # Backpropagation
        loss = loss_X + loss_Y
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if epoch % 40 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    print(f"  Epoch {EPOCHS} | Loss Final: {history[-1]:.6f}")

    # ==========================================================================
    # [FASE 2] O TESTE CRÍTICO DE PLASTICIDADE
    # ==========================================================================
    print("\n[FASE 2] Validação de Plasticidade (Atualização em Tempo Real)")
    print("Ação: Vamos mudar o valor de X para [0.5, 0.5, 0.0, 0.0] na memória.")
    print("Teste: A rede (sem retreino) deve recuperar o NOVO valor imediatamente.")

    # 1. Definir novo valor
    new_val_X = torch.tensor([0.5, 0.5, 0.0, 0.0])

    # 2. Injetar na memória (Single Shot Write)
    with torch.no_grad():
        model("Var_X", new_val_X, is_training_store=True)

    # 3. Fazer a pergunta "cega" novamente
    with torch.no_grad():
        final_pred = model("Var_X", blind_input, is_training_store=False)

    # Resultados
    print("-" * 60)
    print(f"Valor Original de X : {val_X.numpy()}")
    print(f"Novo Valor Injetado : {new_val_X.numpy()}")
    print(f"Resposta da Rede    : {final_pred.numpy().round(2)}")
    print("-" * 60)

    # Verificação Matemática
    mse = torch.mean((new_val_X - final_pred)**2).item()

    if mse < 0.05: # Margem de erro aceitável para redes neurais pequenas
        print("✅ SUCESSO CIENTÍFICO:")
        print("   1. A rede ignorou a entrada vazia.")
        print("   2. A rede navegou no grafo 4D.")
        print("   3. A rede recuperou o estado MAIS RECENTE sem precisar de retreino (Zero-Shot Update).")
    else:
        print("❌ FALHA:")
        print("   A rede não conseguiu acompanhar a atualização da memória.")
        print("   Possível causa: A Fusion Layer ignorou o canal de memória.")

if __name__ == "__main__":
    run_experiment()
