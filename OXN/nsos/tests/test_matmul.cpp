#include "../include/tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

bool is_close(float a, float b, float tol=1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "=== Test Matmul Kernel ===" << std::endl;

    // 1. Identity Test
    // A = [1 2], B = I
    Tensor A({1, 2}, Device::CPU);
    A.data()[0] = 1.0f; A.data()[1] = 2.0f;

    Tensor I = Tensor::eye(2, Device::CPU);
    Tensor C = A.matmul(I);

    if (is_close(C.data()[0], 1.0f) && is_close(C.data()[1], 2.0f)) {
        std::cout << "[PASS] Identity Mul" << std::endl;
    } else {
        std::cout << "[FAIL] Identity Mul: " << C.data()[0] << ", " << C.data()[1] << std::endl;
    }

    // 2. Simple Mul
    // A = [1 1], B = [1 2; 3 4] -> [1*1+1*3, 1*2+1*4] = [4, 6]
    Tensor B({2, 2}, Device::CPU);
    float* b_ptr = B.data();
    b_ptr[0] = 1; b_ptr[1] = 2;
    b_ptr[2] = 3; b_ptr[3] = 4;

    Tensor D = A.matmul(B);
    if (is_close(D.data()[0], 4.0f) && is_close(D.data()[1], 6.0f)) {
        std::cout << "[PASS] 2x2 Mul" << std::endl;
    } else {
        std::cout << "[FAIL] 2x2 Mul: " << D.data()[0] << ", " << D.data()[1] << std::endl;
    }

    // 3. Chrass Layer Simulation (Large Sparse)
    int N = 100;
    Tensor X = Tensor::ones({1, N}, Device::CPU);
    Tensor W = Tensor::eye(N, Device::CPU); // Identity mapping

    Tensor Y = X.matmul(W);

    bool ok = true;
    for(int i=0; i<N; ++i) {
        if (!is_close(Y.data()[i], 1.0f)) ok = false;
    }

    if (ok) std::cout << "[PASS] 100x100 Identity" << std::endl;
    else std::cout << "[FAIL] 100x100 Identity" << std::endl;

    return 0;
}
