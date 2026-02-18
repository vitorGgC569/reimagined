import sys
import os
import torch
import nsos_ext # O módulo C++ compilado

def test_system2_activation():
    print("=== Teste de Ativacao do System 2 (CHRASS Integration) ===")

    # 1. Instanciar Modelo
    print("[1] Instanciando JambaModel (C++)...")
    # Vocab 100, Dim 64, 1 Camada
    model = nsos_ext.JambaModel(1, 64, 100)

    # 2. Criar Input (Batch=1, Seq=10)
    # Criamos uma sequencia com repeticoes para garantir conexoes no grafo
    # Tokens: A B A C B ...
    input_ids = [1, 2, 1, 3, 2, 4, 5, 1, 2, 5]
    tensor_in = nsos_ext.Tensor(input_ids) # Conversao implicita ou manual se necessario
    # O binding espera Tensor input. Vamos checar como 'forward' é exposto.
    # O C++ JambaModel::forward aceita Tensor.
    # Precisamos criar um Tensor a partir de lista ou usar um helper do binding.
    # Assumindo que nsos_ext.Tensor aceita lista ou shape.

    # Vamos tentar criar via binding direto se disponivel, ou mockar se o binding for restrito.
    # Pelo log anterior, "nsos_ext.Tensor" existe.
    # Mas o construtor pode ser apenas Tensor(shape).

    t_in = nsos_ext.Tensor([1, 10, 64]) # Batch, Seq, Dim (Mockado pois forward_embedding espera embeddings ou indices?)
    # forward chama forward_embedding. forward_embedding pega Tensor x.
    # Se x for indices (int), forward_embedding faz lookup.
    # Se x for float (já embeddado), ele usa direto.
    # O codigo C++: forward_embedding(const Tensor& x...)
    #   Tensor h = x; ...
    #   if (needs_reasoning)...
    #   Tensor mem_context = memory->retrieve(h);

    # O codigo assume que 'x' JÁ É o estado latente (embeddings) OU indices?
    # JambaModel::forward_embedding começa com "Tensor h = x".
    # JambaModel::forward chama forward_embedding.
    # JambaModel constructor cria 'embedding'.
    # Mas forward_embedding NÃO PARECE usar 'this->embedding->forward(x)'.
    # Olhando o codigo C++:
    #   Tensor JambaModel::forward(const Tensor& x, Context* ctx) { return forward_embedding(x, ctx); }
    #   Tensor JambaModel::forward_embedding(const Tensor& x, Context* ctx) { Tensor h = x; ... }

    # PARECE QUE O C++ ESPERA JÁ OS EMBEDDINGS NO FORWARD!
    # O D2FDecoder usa model->embedding->forward(seq) antes de chamar model->forward(inputs_3d).
    # Entao devemos passar um Tensor [1, 10, 64] float.

    t_in = nsos_ext.Tensor.random([1, 10, 64]) # Random embeddings

    # 3. Preparar Contexto com Flag
    print("[2] Configurando Contexto com 'force_system2'...")
    ctx = nsos_ext.Context()
    ctx.set_metadata("force_system2", True)

    # 4. Executar Forward
    print("[3] Executando Forward Pass...")
    try:
        out = model.forward(t_in, ctx)
        print("[4] Forward concluido com sucesso.")
        print("   Se você viu logs '[System 2] ...', o teste passou.")
    except Exception as e:
        print(f"[ERRO] Falha no forward: {e}")

if __name__ == "__main__":
    test_system2_activation()
