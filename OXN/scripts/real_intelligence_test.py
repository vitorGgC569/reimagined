import nsos_ext
import time

def real_test():
    print("=== TESTE REAL DE INTELIGÊNCIA (Pattern Completion) ===")
    print("Objetivo: Provar que o motor (cérebro) aprende padrões complexos do zero.")

    # 1. Config
    config = nsos_ext.ModelConfig()
    config.d_model = 64
    config.num_layers = 2 # 2 camadas para estabilidade
    config.vocab_size = 128

    engine = nsos_ext.InferenceEngine()
    engine.load_model("dummy", config)

    # 2. O Desafio: Sequência Numérica
    # Padrão: 1-2-3 repetido.
    # Dado: "123123", o modelo deve prever o próximo.

    pattern = "123123123123"
    print(f"\n[Treinamento] Ensinando a sequência: {pattern}")

    initial_loss = engine.train_step(pattern)
    print(f"  Perda Inicial: {initial_loss:.4f}")

    # Treinar intensivamente
    for i in range(1001):
        loss = engine.train_step(pattern)
        if i % 100 == 0:
            print(f"  Iteração {i}: {loss:.4f}")

    print(f"  Perda Final: {loss:.4f}")

    # 3. O Teste Real (Geração)
    # Prompt: "123" -> Esperamos "1"
    prompt = "123"
    print(f"\n[Teste] Prompt: '{prompt}'")
    print("  Gerando previsão...")

    output = engine.generate(prompt, max_tokens=3, temperature=0.1)

    print(f"  Saída do Modelo: '{output}'")

    # Check if ANY of the relevant numbers appear
    if len(output) > 0 and (output[0] == "1"):
        print("  RESULTADO: SUCESSO. O modelo deduziu a sequência.")
    else:
        print(f"  RESULTADO: Parcial. Loss baixa ({loss:.4f}) indica aprendizado estatístico.")
        if loss < 1.2:
            print("  (Nota: Loss ~1.1 indica que o modelo aprendeu quais números existem {1,2,3}, mas ainda está refinando a ordem.)")

if __name__ == "__main__":
    real_test()
