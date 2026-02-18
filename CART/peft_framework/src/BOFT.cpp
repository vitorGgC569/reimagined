#include "BOFT.h"
#include "LinAlg.h"
#include "TensorOps.h"
#include <iostream>

BOFT::BOFT(int input_dims, int output_dims, int rank)
    : LoRALayer(input_dims, output_dims, rank) {

    // Initialize matrices to be orthogonal from the start
    if (m_A) *m_A = LinAlg::qrDecomposition(*m_A).first;
    if (m_B) *m_B = LinAlg::qrDecomposition(*m_B).first;

    std::cout << "BOFT layer created with orthogonal matrices." << std::endl;
}

Tensor BOFT::forward(const Tensor& input) {
    return LoRALayer::forward(input);
}

void BOFT::orthogonalUpdateStep() {
    std::cout << "Performing orthogonal update step for BOFT..." << std::endl;

    if (!m_grad_A || !m_grad_B) {
        std::cout << "Gradients not available. Skipping update." << std::endl;
        return;
    }

    float learning_rate = 0.01f;

    // 1. Simulate a simple gradient descent step
    for(int i=0; i<m_A->getRows(); ++i) {
        for(int j=0; j<m_A->getCols(); ++j) {
            m_A->at(i,j) -= learning_rate * m_grad_A->at(i,j);
        }
    }
    for(int i=0; i<m_B->getRows(); ++i) {
        for(int j=0; j<m_B->getCols(); ++j) {
            m_B->at(i,j) -= learning_rate * m_grad_B->at(i,j);
        }
    }

    // 2. Re-orthonormalize the matrices to preserve geometry
    // BOFT typically applies this to the columns of the matrices.
    *m_A = LinAlg::qrDecomposition(*m_A).first;
    *m_B = LinAlg::qrDecomposition(*m_B).first;

    std::cout << "BOFT matrices re-orthonormalized." << std::endl;
}
