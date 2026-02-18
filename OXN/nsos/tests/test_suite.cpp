#include "../include/dataloader_v2.h"
#include "../include/fabric.h"
#include "../include/memory_system.h"
#include "../include/sprecher_kan.h"
#include "../include/tensor.h"

using namespace nsos;
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

void test_sprecher() {
  std::cout << "[Test] SprecherKAN..." << std::endl;
  int batch = 2, in = 16, out = 8, hidden = 32;
  SprecherKAN block(in, out, hidden);
  Tensor x = Tensor::random({batch, in}, Device::CPU);
  Tensor y = block.forward(x);

  assert(y.shape[0] == batch);
  assert(y.shape[1] == out);
  // Check values are not NaN/Inf (Basic stability)
  for (int i = 0; i < y.size; ++i) {
    assert(!std::isnan(y.data()[i]));
    assert(!std::isinf(y.data()[i]));
  }
  std::cout << "  Passed." << std::endl;
}

void test_memory_system() {
  std::cout << "[Test] MemorySystem (EpMAN)..." << std::endl;
  int dim = 16;
  MemorySystem mem(dim);

  Tensor key1 = Tensor::ones({1, dim}, Device::CPU);  // Key 1
  Tensor key2 = Tensor::zeros({1, dim}, Device::CPU); // Key 2

  mem.store_episodic(key1);
  mem.store_episodic(key2);

  // assert(mem.episodic_memory.size() == 2); // Internal impl changed to
  // clusters

  // Query close to Key1
  Tensor query = Tensor::ones({1, dim}, Device::CPU);
  Tensor retrieved = mem.retrieve(query);

  // Result should be close to Key1 (1.0) and far from Key2 (0.0)
  // Weighted sum. Sim(q, k1) ~ 1. Sim(q, k2) ~ 0.
  // Exp(1) vs Exp(0) -> 2.71 vs 1.
  // Weight k1 ~ 0.73, Weight k2 ~ 0.27.
  // Result ~ 0.73 * 1 + 0.27 * 0 = 0.73.

  assert(retrieved.data()[0] > 0.5f);
  std::cout << "  Passed." << std::endl;
}

void test_fabric() {
  std::cout << "[Test] Fabric (Ring Reduce)..." << std::endl;
  Fabric fab; // Default constructor auto-detects
  Tensor t = Tensor::ones({10}, Device::CPU);
  fab.all_reduce(t); // Should be no-op or valid
  assert(t.data()[0] == 1.0f);
  std::cout << "  Passed." << std::endl;
}

void test_dataloader() {
  std::cout << "[Test] DataLoader (Mock)..." << std::endl;
  // Create a dummy bin file
  std::string path = "test_data.bin";
  {
    std::vector<float> data(100 * 16, 1.0f);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
  }

  DataLoader loader(path, 10, 16); // Batch 10, Dim 16
  Tensor batch;
  bool success = loader.next(batch);

  assert(success);
  assert(batch.shape[0] == 10);
  assert(batch.shape[1] == 16);
  assert(batch.data()[0] == 1.0f);

  std::cout << "  Passed." << std::endl;
  // Cleanup
  std::remove(path.c_str());
}

int main() {
  std::cout << "=== Running Comprehensive Unit Tests ===" << std::endl;
  test_sprecher();
  test_memory_system();
  test_fabric();
  test_dataloader();
  std::cout << "=== All Unit Tests Passed ===" << std::endl;
  return 0;
}
