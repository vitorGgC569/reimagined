#ifndef HOLOGRAPHIC_H
#define HOLOGRAPHIC_H

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace nsos {

// -- Holographic Associative Memory (HAM) --
// Based on Hyperdimensional Computing (HDC) / Vector Symbolic Architectures
// (VSA). Operations:
// 1. Bind (*): XOR (binary) or Element-wise Mult (real). Combines concepts
// (e.g., Color * Red).
// 2. Bundle (+): Element-wise Add. Creates sets (e.g., Red + Blue).
// 3. Permute (Pi): Rotation. Encodes sequence/order.

class HolographicMemory {
public:
  int get_dim() const { return dim; }
  void to(Device dev);
  Device get_device() const { return item_memory_matrix.get_device(); }
  int dim;
  // We simulate "Continuous" HDC using float vectors (-1, 1 mostly).

  // Store named vectors (Atomic Concepts)
  std::unordered_map<std::string, Tensor> concept_memory;

  // Matrix of all concept vectors [Capacity, Dim] for fast query
  Tensor item_memory_matrix;
  std::vector<std::string> concept_names; // Index to Name mapping
  int capacity = 0;
  int current_size = 0;

  HolographicMemory(int dimension = 10000);

  // Create random hypervector for a concept
  Tensor create_concept(std::string name);

  // Operations
  Tensor bind(const Tensor &A, const Tensor &B);
  Tensor bundle(const std::vector<Tensor> &vectors);
  Tensor permute(const Tensor &A, int shifts = 1);

  // Encoding
  Tensor encode_sequence(const std::vector<std::string> &tokens);

  // Retrieval (Similarity Search)
  // Returns the name of the closest concept in memory
  std::string query(const Tensor &query_vec);

  // Neural Retrieval: Returns a weighted superposition of relevant vectors
  Tensor retrieve_vector(const Tensor &query_vec, int top_k = 5,
                         float temperature = 1.0f);

  // Clean-up memory (normalize to bipolar)
  Tensor clean(const Tensor &noisy_vec);

  // Add known vector (e.g. from external embedding)
  void add_concept(std::string name, const Tensor &vec);
};

} // namespace nsos

#endif
