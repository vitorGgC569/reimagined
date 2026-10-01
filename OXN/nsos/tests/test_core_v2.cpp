#include "../include/nsos_math.h"
#include "../include/nsos_arena.h"
#include "../include/tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace nsos;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_arena() {
    std::cout << "[Test] Arena..." << std::endl;
    void* first = nullptr;
    {
        ArenaScope scope;
        void* p1 = ArenaAllocator::instance().alloc(100, Device::CPU);
        require(p1 != nullptr, "first arena allocation failed");
        void* p2 = ArenaAllocator::instance().alloc(100, Device::CPU);
        require(p2 != nullptr, "second arena allocation failed");
        require(p1 != p2, "arena returned overlapping allocations");
        first = p1;
    }
    {
        ArenaScope scope;
        void* p3 = ArenaAllocator::instance().alloc(100, Device::CPU);
        require(p3 == first,
                "ArenaScope did not rewind the thread-local allocation mark");
    }
    require(ArenaAllocator::instance().alloc(0, Device::CPU) == nullptr,
            "zero-byte arena allocation returned storage");
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
            throw std::runtime_error(
                "MathOps::gemm did not produce the reference value");
        }
    }
    std::cout << "PASS" << std::endl;
}

int main() {
    try {
        test_arena();
        test_math_avx2();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Core v2 test failed: " << error.what() << std::endl;
        return 1;
    }
}
