#include "../include/holographic.h"
#include "../include/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nsos;

// Simple assertion macro
#define ASSERT_TRUE(condition)                                                 \
  if (!(condition)) {                                                          \
    std::cerr << "Assertion failed: " << #condition << " at " << __FILE__      \
              << ":" << __LINE__ << std::endl;                                 \
    std::exit(1);                                                              \
  }

void test_binding_reversibility() {
  std::cout << "[Holo] Testing Binding Reversibility (XOR)..." << std::endl;
  HolographicMemory mem(2000); // 2000 dim for speed
  Tensor A = mem.create_concept("Color");
  Tensor B = mem.create_concept("Red");

  // C = A * B
  Tensor C = mem.bind(A, B);

  // Retrieve B: C * A = (A * B) * A = B * (A * A) = B * 1 (approx)
  Tensor B_recovered = mem.bind(C, A);

  // In Bipolar space, A * A = 1 exactly. So B should be exact.
  std::string match = mem.query(B_recovered);
  std::cout << "Recovered: " << match << std::endl;
  ASSERT_TRUE(match == "Red");
}

void test_bundling_capacity() {
  std::cout << "[Holo] Testing Bundling Capacity..." << std::endl;
  int dim = 10000;
  HolographicMemory mem(dim);
  std::vector<std::string> items = {"Apple", "Banana", "Cherry", "Date",
                                    "Elderberry"};
  std::vector<Tensor> vecs;

  for (const auto &name : items)
    vecs.push_back(mem.create_concept(name));

  Tensor S = mem.bundle(vecs);

  // Check if we can retrieve all items from superposition
  for (const auto &name : items) {
    Tensor t = mem.create_concept(name); // Get existing
    // Check dot product or query
    // Query might be noisy if capacity exceeded, but 5 items in 10k dim is
    // easy. Similarity should be ~ 1/sqrt(K) where K is number of items. S = A
    // + B + ... S . A = A.A + B.A + ... = Dim + Noise.
    std::string res = mem.query(t);
    // Wait, query finds closest concept to S? No, S is close to ALL.
    // query(S) will return one of them (randomly or first).

    // We want to check if 't' is "in" S.
    // We can check dot product.
    float dot = 0;
    for (int i = 0; i < dim; ++i)
      dot += S.data()[i] * t.data()[i];

    // Expected dot: Dim. Noise variance: (K-1)*Dim. StdDev:
    // sqrt(K-1)*sqrt(Dim). Signal-to-Noise Ratio: Dim / (sqrt(K-1)*sqrt(Dim)) =
    // sqrt(Dim / (K-1)). With Dim=10000, K=5, SNR = sqrt(2500) = 50. Huge. Dot
    // should be close to Dim (10000). std::cout << "Dot for " << name << ": "
    // << dot << std::endl;
    ASSERT_TRUE(dot > dim * 0.8f); // Should be very strong
  }
}

void test_sequence_encoding() {
  std::cout << "[Holo] Testing Sequence Encoding..." << std::endl;
  HolographicMemory mem(5000);
  std::vector<std::string> seq = {"The", "quick", "brown", "fox"};
  for (const auto &w : seq)
    mem.create_concept(w);
  mem.create_concept("lazy"); // Distractor

  Tensor S = mem.encode_sequence(seq);

  // S = P0(The) + P1(quick) + P2(brown) + P3(fox)
  // Decode: Who is at Pos 2?
  // Query = permute(S, -2). Expected approx "brown".
  // P-2(S) = P-2(P0(The)) + ... + P-2(P2(brown)) + ...
  //        = P-2(The) + ... + brown + ...

  Tensor Q = mem.permute(S, -2);
  std::string match = mem.query(Q);
  std::cout << "Pos 2 (brown): " << match << std::endl;
  ASSERT_TRUE(match == "brown");

  Tensor Q0 = mem.permute(S, 0); // Pos 0
  std::string match0 = mem.query(Q0);
  std::cout << "Pos 0 (The): " << match0 << std::endl;
  ASSERT_TRUE(match0 == "The");
}

int main() {
  std::cout << "=== HOLOGRAPHIC MEMORY TESTS ===" << std::endl;
  test_binding_reversibility();
  test_bundling_capacity();
  test_sequence_encoding();
  std::cout << "=== ALL TESTS PASSED ===" << std::endl;
  return 0;
}
