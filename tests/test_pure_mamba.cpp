#include "../OXN/nsos/include/mamba2.h"
#include "../OXN/nsos/include/tensor.h"
#include "../OXN/nsos/include/nsos_sdk.h"
#include <iostream>
#include <vector>
#include <cassert>

// Simple mock for context if header is complex, but better to include real one
// Assuming headers are available.

int main() {
    std::cout << "=== Pure C++ Mamba Test ===" << std::endl;

    // Config
    int d_model = 64;
    int d_state = 16;
    int n_heads = 4;
    int batch = 1;
    int seq_len = 50;

    std::cout << "[1] Initializing Mamba2SSD..." << std::endl;
    Mamba2SSD model(d_model, d_state, n_heads);

    std::cout << "[2] Creating Input Tensor..." << std::endl;
    // Input: [Batch, Seq, Dim]
    // Tensor constructor: shape, device, fill
    Tensor input({batch, seq_len, d_model}, Device::CPU, 0.1f);

    Context ctx;

    std::cout << "[3] Running Forward Pass..." << std::endl;
    try {
        Tensor output = model.forward(input, &ctx);
        std::cout << "[4] Forward Complete." << std::endl;

        // Basic Check
        if (output.shape.size() == 3 && output.shape[2] == d_model) {
            std::cout << "SUCCESS: Output shape matches." << std::endl;
        } else {
            std::cout << "FAIL: Output shape mismatch." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "CRITICAL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Test Finished ===" << std::endl;
    return 0;
}
