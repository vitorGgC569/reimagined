#include "LoRA_GA.h"
#include "LinAlg.h"
#include "TensorOps.h"
#include <iostream>

LoRA_GA::LoRA_GA(int input_dims, int output_dims, int rank)
    : LoRALayer(input_dims, output_dims, rank) {}

void LoRA_GA::initializeWithGradients(const Tensor& full_gradient) {
    std::cout << "Initializing LoRA-GA matrices by approximating gradients..." << std::endl;

    // Decompose the gradient using Randomized SVD
    auto [U, S, V_T] = LinAlg::simplifiedSVD(full_gradient, m_rank);

    // Initialize B and A
    // B should be U (or its first 'rank' columns). Our SVD returns U with 'rank' columns.
    // A should be S * V.
    // Dimensions:
    // full_gradient: [out_dim, in_dim]
    // U:             [out_dim, rank]
    // S:             [rank, rank]
    // V:             [rank, in_dim]
    // This matches the required dimensions for m_B and m_A.

    m_B = std::make_unique<Tensor>(U);

    Tensor A_init = TensorOps::multiply(S, V_T);
    m_A = std::make_unique<Tensor>(A_init);

    std::cout << "LoRA-GA matrices initialized successfully." << std::endl;
}
