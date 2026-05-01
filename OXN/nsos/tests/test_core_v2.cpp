#include "../include/nsos_math.h"
#include "../include/nsos_arena.h"
#include "../include/tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace nsos;

void test_arena() {
    std::cout << "[Test] Arena..." << std::endl;
    ArenaAllocator::instance().init(1024 * 1024);
    {
        ArenaScope scope;
        void* p1 = ArenaAllocator::instance().alloc(100, Device::CPU);
        assert(p1 != nullptr);
        void* p2 = ArenaAllocator::instance().alloc(100, Device::CPU);
        assert(p2 != nullptr);
        assert(p1 != p2);
    }
    // Scope exit rewinds offset.
    {
        ArenaScope scope;
        void* p3 = ArenaAllocator::instance().alloc(100, Device::CPU);
        // Should reuse p1's address or similar start
    }
    std::cout << "PASS" << std::endl;
}

void test_math_avx2() {
    std::cout << "[Test] Math AVX2..." << std::endl;
    const int M=6, N=16, K=32;
    std::vector<float> A(M*K, 1.0f);
    std::vector<float> B(K*N, 1.0f);
    std::vector<float> C(M*N, 0.0f);
    
    // C = 1 * A * B
    MathOps::gemm(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
    
    for(float v : C) {
        if(std::abs(v - 32.0f) > 1e-5) {
            std::cout << "FAIL: " << v << " != 32.0" << std::endl;
            return;
        }
    }
    std::cout << "PASS" << std::endl;
}

int main() {
    test_arena();
    test_math_avx2();
    return 0;
}
