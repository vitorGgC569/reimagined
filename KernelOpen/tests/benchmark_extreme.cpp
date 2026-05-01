#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "../src/analog/reram_simulator.h"
#include "../src/quantum/stabilizer_sim.h"
#include "../src/device/quantum_kernel.cuh"
#include "../src/jit/genetic_optimizer.h"

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void stop() { end_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_us() { return std::chrono::duration<double, std::micro>(end_time - start_time).count(); }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
};

void benchmark_phase5_analog() {
    std::cout << "\n[Phase 5] Real Analog Emulation (Noisy ReRAM)" << std::endl;
    uhk::analog::ReRAMCrossbar tile(256, 256);
    std::vector<float> w(256*256, 1.0f);
    tile.program_weights(w);
    std::vector<float> in(256, 0.5f);
    double energy = 0;

    Timer t;
    t.start();
    auto out = tile.compute_analog(in, energy);
    t.stop();

    std::cout << "  Output Current[0]: " << out[0] * 1e6 << " uA (includes noise)" << std::endl;
    std::cout << "  Energy: " << energy * 1e9 << " nJ" << std::endl;
}

void benchmark_phase7_quantum() {
    std::cout << "\n[Phase 7] Real Quantum Stabilizer Simulation" << std::endl;
    uhk::quantum::StabilizerSimulator qsim;

    // Inject Error
    std::cout << "  Injecting X Error on Qubit 4..." << std::endl;
    qsim.inject_error(4, 'X');

    Timer t;
    t.start();

    // 1. Measure (Real Parity Check)
    auto synd = qsim.measure();

    // 2. Decode (Real Lookup)
    uint32_t cx, cz;
    uhk::kernels::decode_syndrome_fast(synd.z_stabilizers, synd.x_stabilizers, &cx, &cz);

    // 3. Correct
    for(int i=0; i<9; ++i) {
        if((cx >> i) & 1) qsim.apply_x_correction(i);
        if((cz >> i) & 1) qsim.apply_z_correction(i);
    }
    t.stop();

    bool clean = qsim.is_logical_clean();
    std::cout << "  Syndrome Z: " << synd.z_stabilizers << " X: " << synd.x_stabilizers << std::endl;
    std::cout << "  Correction Result: " << (clean ? "SUCCESS" : "FAIL") << std::endl;
    std::cout << "  Loop Latency: " << t.elapsed_us() * 1000 << " ns" << std::endl;
}

void benchmark_phase8_jit() {
    std::cout << "\n[Phase 8] Real Self-Writing Kernel (x64 JIT)" << std::endl;
    uhk::jit::GeneticOptimizer optimizer(16, 20); // 16 ops, 20 pop

    int target = 15; // Try to compute '15' using 10, 5
    // Optimal: ADD (10+5) = 15.

    std::cout << "  Target Value: " << target << std::endl;

    for(int g=0; g<20; ++g) {
        optimizer.evolve_generation(target);
        auto best = optimizer.get_best();
        if (best.fitness < 1000) {
            std::cout << "  Generation " << g << " Found Solution! Fitness: " << best.fitness << std::endl;
            break;
        }
    }
}

int main() {
    benchmark_phase5_analog();
    benchmark_phase7_quantum();
    benchmark_phase8_jit();
    return 0;
}
