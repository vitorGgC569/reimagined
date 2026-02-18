#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "../include/LinearModel.hpp"
#include "../include/RingBuffer.hpp"
#include "../include/BitPacking.hpp"
#include "../include/Hilbert.hpp"

using namespace std::chrono;

void print_result(std::string name, size_t n, double seconds) {
    double throughput = (double)n / seconds;
    std::cout << std::left << std::setw(25) << name
              << "| N=" << std::setw(10) << n
              << "| Time: " << std::fixed << std::setprecision(6) << seconds << "s"
              << "| Ops/s: " << std::scientific << std::setprecision(2) << throughput
              << std::defaultfloat << std::endl;
}

void bench_rmi(size_t n) {
    std::vector<double> keys(n);
    std::vector<double> offsets(n);
    for(size_t i=0; i<n; ++i) {
        keys[i] = (double)i;
        offsets[i] = (double)(i * 4); // Linear
    }

    Aion::LinearModel model;

    // Train
    auto start = high_resolution_clock::now();
    model.train(keys, offsets);
    auto end = high_resolution_clock::now();
    print_result("RMI Training", n, duration_cast<duration<double>>(end - start).count());

    // Predict (Scalar)
    start = high_resolution_clock::now();
    double sum = 0;
    for(size_t i=0; i<n; ++i) {
        sum += model.predict(keys[i]);
    }
    end = high_resolution_clock::now();
    print_result("RMI Predict (Scalar)", n, duration_cast<duration<double>>(end - start).count());

    // Predict (Batch/SIMD)
    std::vector<double> results(n);
    start = high_resolution_clock::now();
    model.predict_batch(keys.data(), results.data(), n);
    end = high_resolution_clock::now();
    print_result("RMI Predict (AVX2)", n, duration_cast<duration<double>>(end - start).count());

    // Check sum to prevent opt
    for(auto x : results) sum += x;
    if(sum == 0) std::cout << "";
}

void bench_hilbert(size_t n) {
    auto start = high_resolution_clock::now();
    uint32_t sum = 0;
    int side = 1024; // Grid size
    for(size_t i=0; i<n; ++i) {
        // Simulate random access within grid
        sum += Aion::Hilbert::xy2d(side, i % side, (i * 2) % side);
    }
    auto end = high_resolution_clock::now();
    print_result("Hilbert Mapping", n, duration_cast<duration<double>>(end - start).count());
    if(sum == 0) std::cout << "";
}

void bench_bitpacking(size_t n) {
    std::vector<uint32_t> data(n);
    for(size_t i=0; i<n; ++i) data[i] = i % 31; // 5-bit values

    std::vector<uint64_t> packed((n * 5 + 63) / 64);

    // Scalar
    auto start = high_resolution_clock::now();
    Aion::BitPacker::pack_scalar(data.data(), packed.data(), n, 5);
    auto end = high_resolution_clock::now();
    print_result("BitPacking (Scalar)", n, duration_cast<duration<double>>(end - start).count());

    // AVX-512 (Simulated/Guarded)
    start = high_resolution_clock::now();
    Aion::BitPacker::pack_avx512(data.data(), packed.data(), n, 5);
    end = high_resolution_clock::now();
    print_result("BitPacking (AVX512 Path)", n, duration_cast<duration<double>>(end - start).count());
}

void bench_io(size_t n) {
    Aion::RingBuffer ring(4096);
    char buf[1];

    auto start = high_resolution_clock::now();
    for(size_t i=0; i<n; ++i) {
        ring.submit_read(1, buf, 1, 0, i);
        if (i % 1024 == 0) ring.process_sq(); // Process in batches
    }
    ring.process_sq(); // Finish
    auto end = high_resolution_clock::now();

    print_result("IO_URING (Simulated)", n, duration_cast<duration<double>>(end - start).count());
}

int main() {
    std::cout << ">>> AION C++ Benchmarks (Optimized) <<<" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    std::cout << "\n[Large Scenario]" << std::endl;
    size_t N_LARGE = 10000000; // 10 Million
    bench_rmi(N_LARGE);
    bench_hilbert(N_LARGE);
    bench_bitpacking(N_LARGE);
    bench_io(N_LARGE);

    std::cout << "----------------------------------------------------------------" << std::endl;
    return 0;
}
