#include "MiSS.h"
#include "LinAlg.h"
#include <iostream>
#include <random>
#include <algorithm>

MiSS::MiSS(int input_dims, int output_dims, int initial_rank)
    : LoRALayer(input_dims, output_dims, initial_rank) {}

void MiSS::resizeMatrices(int new_rank) {
    if (new_rank == m_rank) {
        return; // No change needed
    }

    int input_dims = m_A->getCols();
    int output_dims = m_B->getRows();

    auto new_A = std::make_unique<Tensor>(new_rank, input_dims);
    auto new_B = std::make_unique<Tensor>(output_dims, new_rank);

    int rows_to_copy = std::min(m_rank, new_rank);
    int cols_to_copy = std::min(m_rank, new_rank);

    for (int i = 0; i < rows_to_copy; ++i) {
        for (int j = 0; j < input_dims; ++j) {
            new_A->at(i, j) = m_A->at(i, j);
        }
    }
    for (int i = 0; i < output_dims; ++i) {
        for (int j = 0; j < cols_to_copy; ++j) {
            new_B->at(i, j) = m_B->at(i, j);
        }
    }

    m_A = std::move(new_A);
    m_B = std::move(new_B);
    m_rank = new_rank;
}

void MiSS::updateRank(const Tensor& gradient, float threshold) {
    std::cout << "Updating MiSS rank based on gradient importance..." << std::endl;

    // 1. Decompose the gradient to find singular values
    int max_rank = std::min(gradient.getRows(), gradient.getCols());
    auto [U, S, V_T] = LinAlg::simplifiedSVD(gradient, max_rank);

    // 2. Determine new rank based on how many singular values are > threshold
    int new_rank = 0;
    std::cout << "Singular values: ";
    for (int i = 0; i < max_rank; ++i) {
        std::cout << S.at(i, i) << " ";
        if (S.at(i, i) > threshold) {
            new_rank++;
        }
    }
    std::cout << std::endl;
    new_rank = std::max(1, new_rank); // Ensure rank is at least 1

    std::cout << "Old rank: " << m_rank << ". New rank determined by MiSS: " << new_rank << std::endl;

    // 3. Resize matrices to the new rank
    resizeMatrices(new_rank);
}

// Placeholder for the scheduling logic
std::map<int, int> MiSS::createRankSchedule(int num_layers) {
    std::cout << "Creating a dynamic rank allocation schedule for "
              << num_layers << " layers." << std::endl;
    std::map<int, int> schedule;
    for (int i = 0; i < num_layers; ++i) {
        schedule[i] = (i < num_layers / 2) ? 4 : 8;
    }
    return schedule;
}
