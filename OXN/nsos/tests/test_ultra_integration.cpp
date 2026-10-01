#include "mcts_reasoning.h"
#include "memory_system.h"
#include "tensor.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace nsos;

// Simple evaluator for testing
float ultra_evaluator(const Tensor &state) {
    float sum = 0;
    const float* d = state.data();
    for(int i=0; i<state.size; ++i) sum += d[i];
    return sum;
}

void test_autodream_compression() {
    std::cout << "[UltraTest] Testing AutoDream (TurboQuant) Compression..." << std::endl;
    
    MemorySystem mem(128);
    
    // Add 10 clusters to trigger background compression (threshold is 5)
    for (int i = 0; i < 10; ++i) {
        Tensor t = Tensor::zeros({128}, Device::CPU);
        t.data()[0] = (float)i;
        mem.add_cluster(t, "test_cluster_" + std::to_string(i));
    }
    
    std::cout << "[UltraTest] Added 10 clusters. Waiting for AutoDream background thread..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Trigger a retrieval to see if it works with compressed memory
    Tensor query = Tensor::zeros({128}, Device::CPU);
    auto results = mem.retrieve(query, 5);
    
    std::cout << "[UltraTest] Retrieved " << results.size() << " clusters from potentially compressed memory." << std::endl;
    assert(results.size() > 0 && "Should retrieve clusters even after compression");
    
    std::cout << "[SUCCESS] AutoDream logic verified." << std::endl;
}

void test_ultraplan_parallel_mcts() {
    std::cout << "[UltraTest] Testing UltraPlan (Parallel MCTS Agents)..." << std::endl;
    
    Tensor root = Tensor::zeros({10}, Device::CPU);
    MCTSConfig config;
    config.num_simulations = 200;
    config.max_depth = 5;
    
    MCTSReasoning mcts(root, ultra_evaluator, config);
    
    std::cout << "[UltraTest] Spawning 4 parallel UltraPlan agents..." << std::endl;
    // We'll run the parallel search
    mcts.search_ultraplan(4); // 4 agents
    
    int visits = mcts.root_visits();
    std::cout << "[UltraTest] Parallel Search Completed. Root Visits: " << visits << std::endl;
    
    assert(visits >= config.num_simulations && "All simulations should be accounted for");
    
    auto best_path = mcts.get_best_path();
    std::cout << "[UltraTest] Best path depth: " << best_path.size() - 1 << std::endl;
    
    std::cout << "[SUCCESS] UltraPlan parallel orchestration verified." << std::endl;
}

int main() {
    try {
        test_autodream_compression();
        test_ultraplan_parallel_mcts();
        std::cout << "\n[ULTRA] ALL INTEGRATION TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "\n[ULTRA] INTEGRATION TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
