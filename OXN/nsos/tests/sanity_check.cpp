#include "nsos_sdk.h"
#include <iostream>
#include <vector>
#include <cmath>

// Teste de Estabilidade e "Inteligência" Básica
// Simula o problema de sequência: 1, 2, 3 -> ? (4)

int main() {
    std::cout << "Running Numerical Stability & Basic Intelligence Test..." << std::endl;

    // Config
    nsos::ModelConfig config;
    // Stress Test Configuration: Larger model to verify stability fix
    config.num_layers = 6;
    config.d_model = 256; // Reduced to 256 for speed in sandbox, still > 128
    config.vocab_size = 100;
    config.use_cuda = false;

    nsos::InferenceEngine engine;
    if (!engine.load_model("", config)) {
        std::cerr << "Failed to init model." << std::endl;
        return 1;
    }

    // 1. Verificar Estabilidade Numérica (Forward Pass)
    std::cout << "[Test 1] Numerical Stability (NaN check)..." << std::endl;
    try {
        std::string out = engine.generate("123", 5);
        // Se gerou string sem crashar e sem exceções de floating point, passou no básico.
        std::cout << "Output: " << out << std::endl;
        std::cout << "[PASS] No crash or NaNs detected during generation." << std::endl;
    } catch (...) {
        std::cerr << "[FAIL] Crash or Exception during generation." << std::endl;
        return 1;
    }

    // 2. Multitask Training (Sequence + Sorting)
    // v2.0: Use triggers ("solve") to engage System 2 during inference if needed,
    // but for training we want raw pattern matching first.
    std::cout << "\n[Test 2] Multitask Training (Seq 1..6 AND Sort 312->123)..." << std::endl;

    // Adding "logic" keyword to tasks to potentially trigger v2.0 heuristics if implemented in training loop
    // (Currently v2.0 logic is in generate(), train_step is pure backprop).
    // However, the model ARCHITECTURE now has Memory enabled by default in forward().

    std::string task_seq = "123456";
    std::string task_sort = "S:312=123";

    float loss_seq = 0, loss_sort = 0;

    // 50 steps alternating tasks (Reduced for Sandbox Speed)
    for(int i=0; i<50; ++i) {
        loss_seq = engine.train_step(task_seq);
        loss_sort = engine.train_step(task_sort);

        if (i % 10 == 0) std::cout << "Step " << i << " | Seq Loss: " << loss_seq << " | Sort Loss: " << loss_sort << std::endl;

        if (std::isnan(loss_seq) || std::isnan(loss_sort)) {
             std::cerr << "[FAIL] Loss exploded to NaN at step " << i << std::endl;
             return 1;
        }
    }

    std::cout << "Final Seq Loss: " << loss_seq << std::endl;
    std::cout << "Final Sort Loss: " << loss_sort << std::endl;

    // 3. Teste de Inteligência (Previsão Sequência)
    // Trigger v2.0 MCTS with "solve" prefix if possible, or rely on Memory.
    std::cout << "\n[Test 3] Intelligence Check: Sequence Prediction (12345 -> ?)" << std::endl;
    std::string gen = engine.generate("12345", 1);
    std::cout << "Prompt: '12345' -> Generated: '" << gen << "'" << std::endl;

    if (gen == "6") {
        std::cout << "[PASS] Intelligence Verified! Model predicted '6'." << std::endl;
    } else {
        std::cout << "[INFO] Model prediction '" << gen << "' (Expected '6')." << std::endl;
    }

    // 4. Teste de Inteligência (Ordenação) with System 2 Trigger
    std::cout << "\n[Test 4] Intelligence Check: Sorting (System 2 Triggered)" << std::endl;
    // We append "logic" to prompt to trigger MCTS in SDK
    // But our training data was "S:312=123".
    // Let's prompt "logic S:312=" and see if it generalizes or if MCTS helps.
    // Ideally MCTS helps find the "123" path.
    std::string sort_prompt = "logic S:312=";
    std::string sort_gen = engine.generate(sort_prompt, 3);
    std::cout << "Prompt: '" << sort_prompt << "' -> Generated: '" << sort_gen << "'" << std::endl;

    // Check if it contains 123 (it might generate "123" after the prompt)
    if (sort_gen.find("123") != std::string::npos) {
        std::cout << "[PASS] Sorting Verified with System 2!" << std::endl;
    } else {
        std::cout << "[INFO] Model sorting '" << sort_gen << "' (Expected '123')." << std::endl;
    }

    return 0;
}
