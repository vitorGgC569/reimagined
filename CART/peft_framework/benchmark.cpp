#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <memory>

#include "include/Tensor.h"
#include "include/LoRA.h"
#include "include/KAN.h"

// --- Benchmarking Utility ---
void benchmark_layer(const std::string& name, LoRALayer& layer, const Tensor& input) {
    const int num_iterations = 1000;

    auto start_fwd = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        volatile Tensor output = layer.forward(input);
    }
    auto end_fwd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fwd_duration = end_fwd - start_fwd;
    double fwd_ms_per_iter = fwd_duration.count() / num_iterations;

    Tensor upstream_grad(1, input.getCols());
    auto start_bwd = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        layer.backward(upstream_grad);
    }
    auto end_bwd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bwd_duration = end_bwd - start_bwd;
    double bwd_ms_per_iter = bwd_duration.count() / num_iterations;

    std::cout << "--- Benchmark: " << name << " ---" << std::endl;
    std::cout << "Forward Pass:  " << fwd_ms_per_iter << " ms/iteration" << std::endl;
    std::cout << "Backward Pass: " << bwd_ms_per_iter << " ms/iteration" << std::endl;
    std::cout << std::endl;
}


int main() {
    std::cout << "--- Running PEFT Framework Benchmarks ---" << std::endl;

    int in_dim = 128;
    int out_dim = 128;
    int rank = 8;

    Tensor input(1, in_dim);
    Tensor base_weights(out_dim, in_dim);

    LoRALayer lora_baseline(in_dim, out_dim, rank);
    lora_baseline.setBaseWeights(base_weights);
    benchmark_layer("LoRA Baseline", lora_baseline, input);

    return 0;
}
