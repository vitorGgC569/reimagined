#ifndef MEMORY_SYSTEM_H
#define MEMORY_SYSTEM_H

#include "causal_memory_store.h"
#include "oxtamem_ffi.h"
#include "tensor.h"
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "turboquant.h"

namespace nsos {

struct Message {
    std::string role;
    std::string content;
    std::string timestamp;
    std::string agent_id;
};

class MemorySystem {
public:
    struct Cluster {
        Tensor centroid;
        std::vector<Tensor> items;
        std::vector<std::vector<uint8_t>> compressed_items;
        std::chrono::system_clock::time_point last_access;
    };
  MemorySystem(int chunk_dim = 64);

  void store_episodic(const Tensor &state);
  void add_cluster(const Tensor &state, const std::string &label = "");
  void add_instruction(const std::string &instr);
  void add_message(const Message &msg);
  void enable_causal_store(const std::string& path);
  bool enable_oxtamem_store(const std::string& library_path,
                            const std::string& store_path,
                            uint64_t size_mb = 128);
  std::vector<Message> recall_recent_messages(size_t depth) const;

  // Retrieve relevant memory based on query state (Soft Attention simulation)
  Tensor retrieve(const Tensor &query);
  std::vector<Tensor> retrieve(const Tensor &query, size_t top_k);

  // AutoDream: Periodically compress older unstructured memory to quantized blocks
  void run_auto_dream();

  // UltraCompact: Time-based pruning and summarization
  void run_ultra_compact();
  void microcompact_messages();
  void clear_runtime_state();

private:
  void microcompact_messages_locked();
  void validate_state_shape(const Tensor& state, const char* operation) const;

  // Tiered index is private: every access must hold memory_mutex.
  std::vector<Cluster> clusters;
  std::vector<Message> conversation_history;
  mutable std::mutex memory_mutex;
  std::unique_ptr<tq::TurboQuantEngine> tq_engine;
  std::unique_ptr<CausalMemoryStore> causal_store;
  std::unique_ptr<OxtaMemFFI> oxtamem_store;
  std::vector<std::string> instructional_memory;
  int chunk_size;
};

} // namespace nsos

#endif
