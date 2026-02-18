#include "LinAlg.h"
#include <random>
#include <cmath>
#include <iostream>
#include "Tensor.h"
#include "TensorOps.h"

// Note: This file contains simplified linear algebra routines for research purposes.
// For industrial deployment, these should be replaced by calls to optimized libraries
// like Eigen, LAPACK, or cuSOLVER.

namespace LinAlg {

std::tuple<Tensor, Tensor, Tensor> simplifiedSVD(const Tensor& matrix, int rank, int num_iterations) {
    Tensor A = matrix;
    int m = A.getRows();
    int n = A.getCols();

    Tensor U(m, rank);
    Tensor S(rank, rank);
    Tensor V(n, rank);

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int k = 0; k < rank; ++k) {
        Tensor v_k(n, 1);
        for (int i = 0; i < n; ++i) v_k.at(i, 0) = distribution(generator);

        Tensor A_T = A.transpose();
        for (int iter = 0; iter < num_iterations; ++iter) {
            Tensor u_k_unnormalized = TensorOps::multiply(A, v_k);
            Tensor v_k_unnormalized = TensorOps::multiply(A_T, u_k_unnormalized);

            float norm_v = 0;
            for(int i=0; i<v_k_unnormalized.getRows(); ++i) norm_v += v_k_unnormalized.at(i,0) * v_k_unnormalized.at(i,0);
            norm_v = std::sqrt(norm_v);
            if(norm_v > 1e-8)
                for(int i=0; i<v_k_unnormalized.getRows(); ++i) v_k_unnormalized.at(i,0) /= norm_v;

            v_k = v_k_unnormalized;
        }

        Tensor u_k = TensorOps::multiply(A, v_k);
        float sigma_k = 0;
        for (int i = 0; i < u_k.getRows(); ++i) sigma_k += u_k.at(i,0) * u_k.at(i,0);
        sigma_k = std::sqrt(sigma_k);

        if(sigma_k > 1e-8)
            for(int i=0; i<u_k.getRows(); ++i) u_k.at(i,0) /= sigma_k;

        S.at(k, k) = sigma_k;
        for (int i = 0; i < m; ++i) U.at(i, k) = u_k.at(i, 0);
        for (int i = 0; i < n; ++i) V.at(i, k) = v_k.at(i, 0);

        Tensor v_k_T = v_k.transpose();
        Tensor outer_product = TensorOps::multiply(u_k, v_k_T);
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                A.at(i, j) -= sigma_k * outer_product.at(i, j);
            }
        }
    }
    return std::make_tuple(U, S, V.transpose());
}


std::pair<Tensor, Tensor> qrDecomposition(const Tensor& matrix) {
    int m = matrix.getRows();
    int n = matrix.getCols();

    Tensor Q = matrix;
    Tensor R(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            float dot_product = 0;
            for(int k=0; k<m; ++k) dot_product += Q.at(k,i) * Q.at(k,j);

            R.at(j, i) = dot_product;

            for(int k=0; k<m; ++k) Q.at(k,i) -= R.at(j,i) * Q.at(k,j);
        }

        float norm = 0;
        for(int k=0; k<m; ++k) norm += Q.at(k,i) * Q.at(k,i);
        norm = std::sqrt(norm);

        R.at(i, i) = norm;

        if (norm > 1e-8) {
            for(int k=0; k<m; ++k) Q.at(k,i) /= norm;
        }
    }

    return {Q, R};
}

} // namespace LinAlg
