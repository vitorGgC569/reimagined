#include "../include/tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace nsos;

bool is_close(float a, float b, float tol=1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "=== Test Matmul Kernel ===" << std::endl;
    int failures = 0;

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
        ++failures;
    }

    // 2. Simple Mul
    // A = [1 1], B = [1 2; 3 4] -> [1*1+1*3, 1*2+1*4] = [4, 6]
    Tensor A2({1, 2}, Device::CPU);
    A2.data()[0] = 1.0f;
    A2.data()[1] = 1.0f;
    Tensor B({2, 2}, Device::CPU);
    float* b_ptr = B.data();
    b_ptr[0] = 1; b_ptr[1] = 2;
    b_ptr[2] = 3; b_ptr[3] = 4;

    Tensor D = A2.matmul(B);
    if (is_close(D.data()[0], 4.0f) && is_close(D.data()[1], 6.0f)) {
        std::cout << "[PASS] 2x2 Mul" << std::endl;
    } else {
        std::cout << "[FAIL] 2x2 Mul: " << D.data()[0] << ", " << D.data()[1] << std::endl;
        ++failures;
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
    else {
        std::cout << "[FAIL] 100x100 Identity" << std::endl;
        ++failures;
    }

    // 4. Transpose-flag GEMMs must preserve the materialized CPU reference.
    Tensor TN_A({2, 3}, Device::CPU);
    for (int i = 0; i < TN_A.size; ++i) {
        TN_A.data()[i] = static_cast<float>(i + 1) * 0.2f;
    }
    Tensor TN_B({2, 4}, Device::CPU);
    for (int i = 0; i < TN_B.size; ++i) {
        TN_B.data()[i] = static_cast<float>(i - 3) * 0.25f;
    }
    Tensor tn_reference = TN_A.transpose().matmul(TN_B);
    Tensor tn = matmul_tn(TN_A, TN_B);
    for (int i = 0; i < tn.size; ++i) {
        if (tn.data()[i] != tn_reference.data()[i]) ++failures;
    }

    Tensor NT_W({4, 3}, Device::CPU);
    for (int i = 0; i < NT_W.size; ++i) {
        NT_W.data()[i] = static_cast<float>(i + 1) * 0.125f;
    }
    Tensor nt_reference = TN_A.matmul(NT_W.transpose());
    Tensor nt = matmul_nt(TN_A, NT_W);
    for (int i = 0; i < nt.size; ++i) {
        if (nt.data()[i] != nt_reference.data()[i]) ++failures;
    }

    // 5. Device-resident CE API keeps the exact CPU scalar/gradient contract.
    Tensor logits({2, 3}, Device::CPU);
    const float logit_values[] = {1.0f, -0.5f, 0.25f, -1.0f, 2.0f, 0.75f};
    for (int i = 0; i < logits.size; ++i) logits.data()[i] = logit_values[i];
    const std::vector<int> targets = {2, 1};
    const auto ce_reference = logits.cross_entropy(targets);
    const auto ce_device = logits.cross_entropy_device(targets);
    if (ce_device.first.size != 1 ||
        ce_device.first.data()[0] != ce_reference.first) {
        ++failures;
    }
    for (int i = 0; i < ce_reference.second.size; ++i) {
        if (ce_device.second.data()[i] != ce_reference.second.data()[i]) {
            ++failures;
        }
    }

    return failures == 0 ? 0 : 1;
}
