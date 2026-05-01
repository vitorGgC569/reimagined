#include <iostream>
#include <memory>
#include <vector>

#include "BOFT.h"
#include "KAN.h"
#include "LoRA.h"
#include "LoRA_GA.h"
#include "MiSS.h"
#include "Tensor.h"

// Helper function to print a Tensor
void print_tensor(const Tensor& tensor) {
    std::cout << "Tensor (" << tensor.getRows() << "x" << tensor.getCols()
              << ")" << std::endl;
    for (int i = 0; i < tensor.getRows(); ++i) {
        std::cout << "[ ";
        for (int j = 0; j < tensor.getCols(); ++j) {
            std::cout << tensor.at(i, j)
                      << (j == tensor.getCols() - 1 ? "" : ", ");
        }
        std::cout << " ]" << std::endl;
    }
}

int main() {
    std::cout << "--- PEFT Framework Demonstration (Refactored) ---"
              << std::endl;

    // Define model and layer parameters
    int input_dims = 4;
    int output_dims = 4;
    int rank = 2;

    // --- 1. Standard LoRA ---
    std::cout << "\n1. Standard LoRA Layer" << std::endl;
    auto lora = std::make_unique<LoRALayer>(input_dims, output_dims, rank);
    std::cout << "   - LoRA layer created." << std::endl;

    Tensor base_weights(output_dims, input_dims);
    // fill with some values
    for(int i = 0; i < output_dims; ++i)
        for(int j = 0; j < input_dims; ++j)
            base_weights.at(i, j) = (i+j) * 0.1f;
    lora->setBaseWeights(base_weights);

    Tensor input_tensor(1, input_dims);
    input_tensor.at(0,0) = 1.0f; input_tensor.at(0,1) = 2.0f; input_tensor.at(0,2) = 3.0f; input_tensor.at(0,3) = 4.0f;

    std::cout << "   - Input:" << std::endl;
    print_tensor(input_tensor);

    Tensor output_lora = lora->forward(input_tensor);
    std::cout << "   - Output:" << std::endl;
    print_tensor(output_lora);


    // --- 2. LoRA-GA (Gradient Approximation) ---
    std::cout << "\n2. LoRA-GA Layer" << std::endl;
    auto lora_ga =
        std::make_unique<LoRA_GA>(input_dims, output_dims, rank);
    Tensor dummy_gradient(output_dims, input_dims);
    lora_ga->initializeWithGradients(dummy_gradient);
    std::cout << "   - LoRA-GA layer initialized." << std::endl;

    // --- 3. MiSS (Multiple-stage importance-aware Sparse Sharing) ---
    std::cout << "\n3. MiSS Layer" << std::endl;
    int initial_rank = 2;
    auto miss =
        std::make_unique<MiSS>(input_dims, output_dims, initial_rank);
    std::cout << "   - MiSS layer created with initial rank " << initial_rank
              << "." << std::endl;
    // Demonstrate rank update with a dummy gradient
    Tensor dummy_miss_grad(output_dims, input_dims);
    miss->updateRank(dummy_miss_grad);
    std::cout << "   - MiSS layer rank updated based on dummy gradient." << std::endl;

    // --- 4. BOFT (Butterfly Orthogonal Fine-Tuning) ---
    std::cout << "\n4. BOFT Layer" << std::endl;
    auto boft = std::make_unique<BOFT>(input_dims, output_dims, rank);
    std::cout << "   - BOFT layer created." << std::endl;
    boft->orthogonalUpdateStep();
    std::cout << "   - BOFT orthogonal update step simulated." << std::endl;

    // --- 5. KAN (Kolmogorov-Arnold Network) Layer ---
    // Note: KAN refactoring will happen in the next phase.
    // This part is temporarily commented out to allow compilation.
    /*
    std::cout << "\n5. KAN Layer" << std::endl;
    int kan_input_dims = 3;
    int kan_output_dims = 2;
    int spline_order = 3;
    KANLayer kan(kan_input_dims, kan_output_dims, spline_order);
    std::cout << "   - KAN layer created with input_dims=" << kan_input_dims
              << ", output_dims=" << kan_output_dims
              << ", spline_order=" << spline_order << "." << std::endl;
    std::vector<float> kan_input = {0.5f, -0.2f, 1.3f};
    std::vector<float> kan_output = kan.forward(kan_input);
    */

    std::cout << "\n--- Demonstration Complete ---" << std::endl;

    return 0;
}
