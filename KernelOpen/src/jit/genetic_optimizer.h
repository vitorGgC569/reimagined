#ifndef UHK_JIT_GENETIC_OPTIMIZER_H
#define UHK_JIT_GENETIC_OPTIMIZER_H

#include <vector>
#include <algorithm>
#include <random>
#include "../jit/x64_jit_compiler.h"

namespace uhk {
namespace jit {

// =============================================================================
// Phase 8: Real Self-Writing Kernel (Genetic Optimizer)
// =============================================================================
// Manages a population of x86 machine code sequences.
// Executes them FOR REAL using X64JIT to measure performance.
// =============================================================================

struct KernelGenome {
    std::vector<uint8_t> opcodes;
    double fitness; // Execution time (lower is better)
};

class GeneticOptimizer {
public:
    GeneticOptimizer(int genome_size, int pop_size) : jit_engine_() {
        rng_.seed(1337);
        for (int i = 0; i < pop_size; ++i) {
            KernelGenome g;
            g.opcodes.resize(genome_size);
            for (auto& op : g.opcodes) op = rand() % 4; // Valid: 0..3
            g.fitness = 1e9;
            population_.push_back(g);
        }
    }

    void evolve_generation(int64_t target_val) {
        // 1. Evaluate Fitness (Run the code!)
        int64_t data[2] = {10, 5}; // Inputs

        for (auto& ind : population_) {
            // JIT Compile
            auto func = jit_engine_.compile(ind.opcodes);

            // Measure Execution Time (Real CPU Cycles)
            // Using rdtsc for high precision
            unsigned int dum;
            uint64_t start = __builtin_ia32_rdtscp(&dum);

            // EXECUTE GENERATED MACHINE CODE
            data[0] = 10; data[1] = 5; // Reset inputs
            func(data);

            uint64_t end = __builtin_ia32_rdtscp(&dum);

            // Fitness: Distance from target value + Cycle count penalty
            // We want it correct AND fast.
            int64_t result = data[0];
            double dist = std::abs(result - target_val);
            double cycles = (double)(end - start);

            ind.fitness = (dist * 1000.0) + cycles;
        }

        // 2. Selection (Sort by fitness)
        std::sort(population_.begin(), population_.end(),
            [](const KernelGenome& a, const KernelGenome& b) { return a.fitness < b.fitness; });

        // 3. Crossover & Mutation
        int cutoff = population_.size() / 2;
        for (int i = cutoff; i < population_.size(); ++i) {
            // Clone from elite
            population_[i] = population_[i % cutoff];

            // Mutate instruction
            int mutation_idx = rand() % population_[i].opcodes.size();
            population_[i].opcodes[mutation_idx] = rand() % 4;
        }
    }

    KernelGenome get_best() { return population_[0]; }

private:
    std::vector<KernelGenome> population_;
    std::mt19937 rng_;
    X64JIT jit_engine_;
};

} // namespace jit
} // namespace uhk

#endif // UHK_JIT_GENETIC_OPTIMIZER_H
