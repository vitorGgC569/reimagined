#include "nsos_sdk.h"
#include <iostream>
#include <chrono>

// Exemplo de Marketing: Inferência na Borda (Raspberry Pi 4/5)
// Foco: Baixo consumo de memória (BitNet) e Velocidade

int main() {
    std::cout << "=== NSOS Edge Inference Demo (Raspberry Pi Optimized) ===" << std::endl;

    // 1. Configuração "Edge" (Quantização Extrema)
    nsos::ModelConfig config;
    // Matching config from mock_llama_factory.py
    config.num_layers = 4;
    config.d_model = 128;
    config.vocab_size = 100;
    config.use_quantization = true;
    config.use_cuda = false;

    // 2. Inicialização
    nsos::InferenceEngine engine;
    auto start_load = std::chrono::high_resolution_clock::now();

    // Load from generated mock weights
    if (engine.load_model("mock_llama_weights", config)) {
        std::cout << "[SUCCESS] Model loaded from mock_llama_weights." << std::endl;
    } else {
        std::cout << "[INFO] Running with Random Initialization (Demo Mode)" << std::endl;
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> load_dur = end_load - start_load;
    std::cout << "Load Time: " << load_dur.count() << "s | RAM Usage: " << engine.get_memory_usage() << " MB" << std::endl;

    // 3. Inferência em Loop
    std::string prompts[] = {
        "Hello, how are you?",
        "Translate to Spanish: The cat is on the table.",
        "Write a python function to sum two numbers."
    };

    for (const auto& p : prompts) {
        std::cout << "\n> User: " << p << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string reply = engine.generate(p, 64);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> gen_dur = t1 - t0;
        float tps = 64.0f / gen_dur.count(); // Aproximado

        std::cout << "> NSOS: " << reply << std::endl;
        std::cout << "[Stats] Speed: " << tps << " tok/s (CPU)" << std::endl;
    }

    return 0;
}
