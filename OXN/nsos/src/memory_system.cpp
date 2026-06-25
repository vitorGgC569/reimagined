#include "memory_system.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace nsos {

namespace {

constexpr std::array<char, 4> kMessageFormatMagic{{'N', 'S', 'M', '1'}};
constexpr float kMaxMemorySimilarityLogit = 40.0f;

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(value));
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("Corrupted message payload");
  }
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  offset += sizeof(value);
  return value;
}

void append_string_field(std::vector<uint8_t>& out, const std::string& value) {
  if (value.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Message field too large to serialize");
  }
  append_u32(out, static_cast<uint32_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
}

std::string read_string_field(const std::vector<uint8_t>& bytes, size_t& offset) {
  const uint32_t length = read_u32(bytes, offset);
  if (offset + length > bytes.size()) {
    throw std::runtime_error("Corrupted message payload");
  }
  std::string value(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                    bytes.begin() + static_cast<std::ptrdiff_t>(offset + length));
  offset += length;
  return value;
}

float stable_similarity_weight(float dot) {
  if (!std::isfinite(dot)) {
    return 0.0f;
  }
  return std::exp(std::clamp(dot, -kMaxMemorySimilarityLogit, kMaxMemorySimilarityLogit));
}

std::vector<uint8_t> serialize_message(const Message& msg) {
  std::vector<uint8_t> bytes;
  bytes.reserve(4 + 4 * sizeof(uint32_t) + msg.role.size() + msg.agent_id.size() +
                msg.timestamp.size() + msg.content.size());
  bytes.insert(bytes.end(), kMessageFormatMagic.begin(), kMessageFormatMagic.end());
  append_string_field(bytes, msg.role);
  append_string_field(bytes, msg.agent_id);
  append_string_field(bytes, msg.timestamp);
  append_string_field(bytes, msg.content);
  return bytes;
}

Message deserialize_message(const std::vector<uint8_t>& bytes) {
  if (bytes.size() >= kMessageFormatMagic.size() &&
      std::equal(kMessageFormatMagic.begin(), kMessageFormatMagic.end(), bytes.begin())) {
    size_t offset = kMessageFormatMagic.size();
    Message message;
    message.role = read_string_field(bytes, offset);
    message.agent_id = read_string_field(bytes, offset);
    message.timestamp = read_string_field(bytes, offset);
    message.content = read_string_field(bytes, offset);
    return message;
  }

  std::string flat(bytes.begin(), bytes.end());
  Message message;
  size_t first = flat.find('\n');
  size_t second = first == std::string::npos ? std::string::npos : flat.find('\n', first + 1);
  size_t third = second == std::string::npos ? std::string::npos : flat.find('\n', second + 1);
  if (first == std::string::npos || second == std::string::npos || third == std::string::npos) {
    message.content = flat;
    return message;
  }
  message.role = flat.substr(0, first);
  message.agent_id = flat.substr(first + 1, second - first - 1);
  message.timestamp = flat.substr(second + 1, third - second - 1);
  message.content = flat.substr(third + 1);
  return message;
}

std::vector<uint8_t> serialize_tensor_state(const Tensor& state) {
  Tensor host_state = (state.get_device() == Device::GPU) ? state.cpu() : state;
  const uint32_t rank = static_cast<uint32_t>(host_state.shape.size());
  const size_t header_bytes = sizeof(rank) + rank * sizeof(int32_t);
  const size_t data_bytes = static_cast<size_t>(host_state.size) * sizeof(float);
  std::vector<uint8_t> bytes(header_bytes + data_bytes);
  uint8_t* out = bytes.data();
  std::memcpy(out, &rank, sizeof(rank));
  out += sizeof(rank);
  for (int dim : host_state.shape.dims) {
    int32_t dim32 = dim;
    std::memcpy(out, &dim32, sizeof(dim32));
    out += sizeof(dim32);
  }
  std::memcpy(out, host_state.data(), data_bytes);
  return bytes;
}

} // namespace

MemorySystem::MemorySystem(int chunk_dim) : chunk_size(chunk_dim) {}

void MemorySystem::add_cluster(const Tensor &state, const std::string &label) {
  (void)label;
  store_episodic(state);
}

void MemorySystem::enable_causal_store(const std::string& path) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  causal_store = std::make_unique<CausalMemoryStore>(path);
}

bool MemorySystem::enable_oxtamem_store(const std::string& library_path,
                                        const std::string& store_path,
                                        uint64_t size_mb) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  auto backend = std::make_unique<OxtaMemFFI>();
  if (!backend->load(library_path) || !backend->open(store_path, size_mb)) {
    return false;
  }
  oxtamem_store = std::move(backend);
  return true;
}

void MemorySystem::store_episodic(const Tensor &state) {
  std::lock_guard<std::mutex> lock(memory_mutex);

  // Initialize cluster's access time
  auto now = std::chrono::system_clock::now();

  // The centroid math below dereferences host pointers, so the incoming
  // state (which may live on GPU) must be materialized on the host first.
  // Without this, state.data() returns a device pointer that is read on
  // the CPU -> undefined behavior / segfault.  The clones stored in the
  // cluster preserve the original device so retrieval semantics and
  // checkpoint contents are unchanged for CPU callers.
  Tensor state_cpu = (state.get_device() == Device::GPU) ? state.cpu() : state;

  // Clustered episodic storage with centroid routing.
  float best_dist = 1e9;
  int best_cluster = -1;

  // Find best cluster
  for (size_t i = 0; i < clusters.size(); ++i) {
    // Euclidean distance to centroid (simplistic).  Guard against a size
    // mismatch between this state and a previously stored centroid (states
    // of differing dimensionality must never index past either buffer).
    Tensor centroid_cpu = clusters[i].centroid.get_device() == Device::GPU
                              ? clusters[i].centroid.cpu()
                              : clusters[i].centroid;
    if (centroid_cpu.size != state_cpu.size) {
      continue;
    }
    float dist = 0;
    const float *c = centroid_cpu.data();
    const float *s = state_cpu.data();
    for (int k = 0; k < state_cpu.size; ++k)
      dist += (c[k] - s[k]) * (c[k] - s[k]);

    if (dist < best_dist) {
      best_dist = dist;
      best_cluster = (int)i;
    }
  }

  // Threshold to create new cluster
  float CLUSTER_RADIUS = 10.0f; // Tunable
  if (best_cluster == -1 || best_dist > CLUSTER_RADIUS) {
    // Create new
    Cluster c;
    c.centroid = state.clone(); // Clone
    c.items.push_back(state.clone());
      c.last_access = now;
      clusters.push_back(c);
  } else {
    // Add to existing
    clusters[best_cluster].items.push_back(state.clone());
    clusters[best_cluster].last_access = now;
    // Update centroid (Moving Average).  Operate on a host copy then write
    // the result back onto the centroid in its original device, keeping the
    // host-pointer arithmetic safe for GPU-resident centroids.
    float alpha = 0.1f;
    const Device centroid_device = clusters[best_cluster].centroid.get_device();
    Tensor centroid_cpu = centroid_device == Device::GPU
                              ? clusters[best_cluster].centroid.cpu()
                              : clusters[best_cluster].centroid;
    if (centroid_cpu.size == state_cpu.size) {
      float *c = centroid_cpu.data();
      const float *s = state_cpu.data();
      for (int k = 0; k < state_cpu.size; ++k)
        c[k] = (1 - alpha) * c[k] + alpha * s[k];
      clusters[best_cluster].centroid =
          centroid_device == Device::GPU ? centroid_cpu.to(centroid_device)
                                         : centroid_cpu;
    }
  }

  if (causal_store) {
    causal_store->append("episodic", serialize_tensor_state(state));
  }
  if (oxtamem_store) {
    oxtamem_store->write("episodic", serialize_tensor_state(state));
  }
}

void MemorySystem::add_instruction(const std::string &instr) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  instructional_memory.push_back(instr);
}

Tensor MemorySystem::retrieve(const Tensor &query) {
  std::lock_guard<std::mutex> lock(memory_mutex);

  Tensor query_cpu = (query.get_device() == Device::GPU) ? query.cpu() : query;
  const float *query_ptr = query_cpu.data();
  if (clusters.empty() || query_cpu.size == 0) {
    return Tensor::zeros(query_cpu.shape.dims, Device::CPU).to(query.get_device());
  }

  std::vector<std::pair<float, int>> centroid_scores;
  centroid_scores.reserve(clusters.size());
  float query_norm_sq = 0.0f;
  for (int i = 0; i < query_cpu.size; ++i) {
    query_norm_sq += query_ptr[i] * query_ptr[i];
  }
  const float query_norm = std::sqrt(std::max(query_norm_sq, 1e-8f));

  for (size_t i = 0; i < clusters.size(); ++i) {
    float dot = 0.0f;
    float centroid_norm_sq = 0.0f;
    Tensor centroid_cpu =
        clusters[i].centroid.get_device() == Device::GPU ? clusters[i].centroid.cpu()
                                                         : clusters[i].centroid;
    const float *c = centroid_cpu.data();
    // Centroids of a different dimensionality than the query must not be
    // indexed past their buffer; only the overlapping prefix contributes.
    const int centroid_extent = std::min(query_cpu.size, centroid_cpu.size);
    for (int k = 0; k < centroid_extent; ++k) {
      dot += c[k] * query_ptr[k];
      centroid_norm_sq += c[k] * c[k];
    }
    const float centroid_norm = std::sqrt(std::max(centroid_norm_sq, 1e-8f));
    const float cosine = dot / (query_norm * centroid_norm);
    centroid_scores.push_back({cosine, static_cast<int>(i)});
  }

  const size_t shortlist =
      std::min<size_t>(std::max<size_t>(1, std::min<size_t>(clusters.size(), 8)), centroid_scores.size());
  std::partial_sort(
      centroid_scores.begin(),
      centroid_scores.begin() + shortlist,
      centroid_scores.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

  std::vector<int> relevant_clusters;
  relevant_clusters.reserve(shortlist);
  for (size_t idx = 0; idx < shortlist; ++idx) {
    relevant_clusters.push_back(centroid_scores[idx].second);
    clusters[static_cast<size_t>(centroid_scores[idx].second)].last_access =
        std::chrono::system_clock::now();
  }

  Tensor context_cpu = Tensor::zeros(query_cpu.shape.dims, Device::CPU);
  float total_weight = 0.0f;

  std::vector<float> query_vec;
  if (!clusters.empty()) {
     query_vec.assign(query_ptr, query_ptr + query_cpu.size);
  }

  for (int idx : relevant_clusters) {
    // Process uncompressed items
    for (const auto &mem : clusters[idx].items) {
      Tensor mem_cpu = mem.get_device() == Device::GPU ? mem.cpu() : mem;
      float dot = 0;
      const float *mem_ptr = mem_cpu.data();
      // The query, the stored item, and the accumulation buffer may differ
      // in length; clamp every access to their common extent so neither
      // query_ptr, mem_ptr nor context_cpu is indexed out of bounds.
      const int mem_extent =
          std::min(mem_cpu.size, std::min(query_cpu.size, context_cpu.size));
      for (int i = 0; i < mem_extent; ++i)
        dot += query_ptr[i] * mem_ptr[i];

      float weight = stable_similarity_weight(dot);

      for (int i = 0; i < mem_extent; ++i)
        context_cpu.data()[i] += mem_ptr[i] * weight;
      total_weight += weight;
    }

    // Process compressed items via TurboQuant Dot
    if (tq_engine && !clusters[idx].compressed_items.empty()) {
        for (const auto &compressed_mem : clusters[idx].compressed_items) {
            float dot = tq_engine->dot(query_vec, compressed_mem);
            float weight = stable_similarity_weight(dot);

            // Decode entirely to reconstruct the state as we need it for weighting.
            // The decoded vector length need not match the accumulation buffer,
            // so clamp to the smaller extent to avoid writing past context_cpu.
            std::vector<float> decoded = tq_engine->decode(compressed_mem);
            const size_t decoded_extent =
                std::min(decoded.size(), static_cast<size_t>(context_cpu.size));
            for (size_t i = 0; i < decoded_extent; ++i) {
                context_cpu.data()[i] += decoded[i] * weight;
            }
            total_weight += weight;
        }
    }
  }

  if (std::isfinite(total_weight) && total_weight > 1e-6f) {
    for (int i = 0; i < context_cpu.size; ++i)
      context_cpu.data()[i] /= total_weight;
  }

  return context_cpu.to(query.get_device());
}

std::vector<Tensor> MemorySystem::retrieve(const Tensor &query, size_t top_k) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  std::vector<std::pair<float, Tensor>> ranked;
  ranked.reserve(clusters.size());

  Tensor query_cpu = (query.get_device() == Device::GPU) ? query.cpu() : query;
  const float *query_ptr = query_cpu.data();

  for (const auto &cluster : clusters) {
    Tensor centroid_cpu =
        cluster.centroid.get_device() == Device::GPU ? cluster.centroid.cpu()
                                                     : cluster.centroid;
    const float *centroid_ptr = centroid_cpu.data();
    float dot = 0.0f;
    const int extent = std::min(query_cpu.size, centroid_cpu.size);
    for (int i = 0; i < extent; ++i) {
      dot += query_ptr[i] * centroid_ptr[i];
    }
    ranked.push_back({dot, cluster.centroid});
  }

  std::partial_sort(
      ranked.begin(),
      ranked.begin() + std::min(top_k, ranked.size()),
      ranked.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::vector<Tensor> results;
  const size_t limit = std::min(top_k, ranked.size());
  results.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    results.push_back(ranked[i].second);
  }
  return results;
}

void MemorySystem::run_auto_dream() {
  std::lock_guard<std::mutex> lock(memory_mutex);
  if (clusters.empty()) return;

  // Initialize engine precisely
  if (!tq_engine && !clusters[0].items.empty()) {
      int dim = clusters[0].items[0].size;
      if (dim % 2 == 0 && dim > 0) {
          tq_engine = std::make_unique<tq::TurboQuantEngine>(dim);
      }
  }

  if (!tq_engine) return;

  for (auto& cluster : clusters) {
      if (cluster.items.size() > 5) { 
          std::vector<std::vector<uint8_t>> encoded_items;
          encoded_items.reserve(cluster.items.size());
          bool all_encoded = true;
          for (const auto& tensor : cluster.items) {
              Tensor host_tensor =
                  tensor.get_device() == Device::GPU ? tensor.cpu() : tensor;
              std::vector<float> flat_data(host_tensor.data(),
                                           host_tensor.data() + host_tensor.size);
              try {
                  encoded_items.push_back(tq_engine->encode(flat_data));
              } catch (const std::exception& error) {
                  all_encoded = false;
                  std::cerr << "[MemorySystem] AutoDream encode failed: "
                            << error.what() << "\n";
                  break;
              }
          }
          if (all_encoded) {
              cluster.compressed_items.insert(cluster.compressed_items.end(),
                                              encoded_items.begin(),
                                              encoded_items.end());
              cluster.items.clear();
          }
      }
  }
}

void MemorySystem::add_message(const Message &msg) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    conversation_history.push_back(msg);
    microcompact_messages();
    if (causal_store) {
        causal_store->append("messages", serialize_message(msg));
    }
    if (oxtamem_store) {
        oxtamem_store->write("messages", serialize_message(msg));
    }
}

std::vector<Message> MemorySystem::recall_recent_messages(size_t depth) const {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<Message> messages;

    if (oxtamem_store) {
        for (const auto& payload : oxtamem_store->recall("messages", depth)) {
            messages.push_back(deserialize_message(payload));
        }
        if (!messages.empty()) {
            return messages;
        }
    }

    if (causal_store) {
        for (const auto& payload : causal_store->read_history("messages", depth)) {
            messages.push_back(deserialize_message(payload));
        }
    }
    if (!messages.empty()) {
        return messages;
    }
    if (depth == 0 || conversation_history.empty()) {
        return {};
    }
    const size_t begin =
        conversation_history.size() > depth ? conversation_history.size() - depth : 0;
    messages.insert(messages.end(),
                    conversation_history.begin() + static_cast<std::ptrdiff_t>(begin),
                    conversation_history.end());
    return messages;
}

void MemorySystem::microcompact_messages() {
    if (conversation_history.size() > 20) {
        // Snipping oldest messages
        conversation_history.erase(conversation_history.begin(), conversation_history.begin() + 10);
        std::cout << "[MicroCompact] Trimmed oldest messages.\n";
    }
}

void MemorySystem::run_ultra_compact() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    if (!tq_engine || clusters.empty()) return;
    
    auto now = std::chrono::system_clock::now();
    for (auto& cluster : clusters) {
        auto age = std::chrono::duration_cast<std::chrono::minutes>(now - cluster.last_access).count();
        if (age > 10 && !cluster.compressed_items.empty()) {
            std::cout << "[UltraCompact] Performing Weighted Rollup (Node Sniping) for stale cluster...\n";
            
            // 1. Decode all knowledge in this cluster
            std::vector<std::vector<float>> decoded_items;
            for (const auto& comp : cluster.compressed_items) {
                try {
                    decoded_items.push_back(tq_engine->decode(comp));
                } catch (const std::exception& error) {
                    std::cerr << "[MemorySystem] UltraCompact decode failed: "
                              << error.what() << "\n";
                }
            }
            
            if (decoded_items.empty()) continue;

            // 2. Compute Weighted Average (Centroid Rollup)
            size_t dim = decoded_items[0].size();
            std::vector<float> rollup(dim, 0.0f);
            for (const auto& item : decoded_items) {
                for (size_t i = 0; i < dim; ++i) rollup[i] += item[i];
            }
            for (size_t i = 0; i < dim; ++i) rollup[i] /= decoded_items.size();

            // 3. Re-quantize as new Consolidated Centroid
            try {
                auto new_centroid_compressed = tq_engine->encode(rollup);
                cluster.compressed_items.clear();
                cluster.compressed_items.push_back(new_centroid_compressed);
                
                // Update the floating-point centroid tensor too
                float* c_data = cluster.centroid.data();
                for (size_t i = 0; i < dim; ++i) c_data[i] = rollup[i];
                
                std::cout << "[UltraCompact] Cluster consolidated into 1 summarized state.\n";
            } catch (const std::exception& error) {
                std::cerr << "[MemorySystem] UltraCompact encode failed: "
                          << error.what() << "\n";
            }
        }
    }
}

void MemorySystem::clear_runtime_state() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    clusters.clear();
    conversation_history.clear();
    instructional_memory.clear();
    tq_engine.reset();
}

} // namespace nsos
