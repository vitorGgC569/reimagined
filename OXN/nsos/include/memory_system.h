#ifndef MEMORY_SYSTEM_H
#define MEMORY_SYSTEM_H

#include "tensor.h"
#include <mutex>
#include <string>
#include <vector>

namespace nsos {

class MemorySystem {
public:
  struct Cluster {
    Tensor centroid;
    std::vector<Tensor> items;
  };
  std::vector<Cluster> clusters;   // Tiered Index
  mutable std::mutex memory_mutex; // Mutex for thread safety

  // Rhea: Instructional Memory (High fidelity constraints)
  std::vector<std::string> instructional_memory;

  int chunk_size;

  MemorySystem(int chunk_dim = 64);

  void store_episodic(const Tensor &state);
  void add_instruction(const std::string &instr);

  // Retrieve relevant memory based on query state (Soft Attention simulation)
  Tensor retrieve(const Tensor &query);
};

} // namespace nsos

#endif
