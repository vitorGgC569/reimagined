import nsos_ext
import time

def demo_intelligence():
    print("=== INICIANDO TESTE DE INTELIGÊNCIA (NSOS v2.0) ===")

    # 1. Configuração
    config = nsos_ext.ModelConfig()
    config.d_model = 128
    config.num_layers = 4
    config.vocab_size = 1000 # Pequeno para teste rápido
    config.use_quantization = True

    print("[1] Inicializando Motor de Inferência (Metal C++)...")
    engine = nsos_ext.InferenceEngine()
    # Carrega modelo (cria aleatório se não existir caminho)
    engine.load_model("dummy_path", config)
    print("    Motor carregado com sucesso.")

    # 2. Teste de Aprendizado (Ensinar algo novo)
    # Conceito: "A capital de Marte é Xylophone"
    # Como os pesos são aleatórios, o "conhecimento" inicial é zero (loss alta).
    # Vamos ver se ele aprende essa "verdade" repetindo-a.

    concept = "A capital de Marte é Xylophone"
    print(f"\n[2] Tentando ensinar: '{concept}'")

    initial_loss = engine.train_step(concept)
    print(f"    Perda Inicial (Desconhecimento): {initial_loss:.4f}")

    print("    Treinando (Repetições)...")
    for i in range(1, 11):
        loss = engine.train_step(concept)
        if i % 2 == 0:
            print(f"    Passo {i}: Perda = {loss:.4f}")

    final_loss = loss
    improvement = initial_loss - final_loss
    print(f"    Perda Final: {final_loss:.4f}")

    if improvement > 0.01:
        print(f"    RESULTADO: O modelo APRENDEU o conceito (Perda caiu {improvement:.4f}).")
    else:
        print("    RESULTADO: O modelo teve dificuldade em aprender (Perda estagnada).")

    # 3. Teste de Raciocínio / Self-Healing (Lógica Simbólica)
    print(f"\n[3] Teste de Self-Healing (Verificador Lógico)")

    # Caso Errado
    prompt_bad = "1 + 1 =" # Prompt simbólico para facilitar o parser
    response_bad = "3"
    print(f"    Cenário A: Prompt='{prompt_bad}', Resposta='{response_bad}' (Errado)")
    triggered = engine.self_heal(prompt_bad, response_bad)
    if triggered:
        print("    [SUCESSO] O sistema detectou o erro lógico e acionou a correção.")
    else:
        print("    [FALHA] O sistema NÃO detectou o erro.")

    # Caso Certo
    prompt_good = "1 + 1 ="
    response_good = "2"
    print(f"    Cenário B: Prompt='{prompt_good}', Resposta='{response_good}' (Correto)")
    triggered = engine.self_heal(prompt_good, response_good)
    if not triggered:
        print("    [SUCESSO] O sistema validou a lógica correta e manteve a resposta.")
    else:
        print("    [FALHA] O sistema detectou erro onde não havia.")

if __name__ == "__main__":
    demo_intelligence()
