#include "memory_system.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <set>

namespace nsos {

namespace {

constexpr std::array<char, 4> kMessageFormatMagic{{'N', 'S', 'M', '1'}};
constexpr std::array<char, 4> kTensorFormatMagic{{'N', 'S', 'T', '1'}};
constexpr size_t kOxtaVectorDimensions = 128;
constexpr size_t kMaxMessageFieldBytes = 4 * 1024 * 1024;
constexpr size_t kMaxInstructionBytes = 1024 * 1024;
constexpr size_t kMaxInstructions = 10000;
constexpr size_t kMaxClusters = 4096;
constexpr size_t kMaxItemsPerCluster = 1024;
constexpr size_t kMaxRuntimeItems = 16384;
constexpr float kClusterRadiusSquared = 10.0f;

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  for (size_t byte = 0; byte < sizeof(value); ++byte) {
    out.push_back(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
  }
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("Corrupted message payload");
  }
  uint32_t value = 0;
  for (size_t byte = 0; byte < sizeof(value); ++byte) {
    value |= static_cast<uint32_t>(bytes[offset + byte]) << (byte * 8);
  }
  offset += sizeof(value);
  return value;
}

void append_string_field(std::vector<uint8_t>& out, const std::string& value) {
  if (value.size() > kMaxMessageFieldBytes) {
    throw std::runtime_error("Message field too large to serialize");
  }
  append_u32(out, static_cast<uint32_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
}

std::string read_string_field(const std::vector<uint8_t>& bytes, size_t& offset) {
  const uint32_t length = read_u32(bytes, offset);
  if (length > kMaxMessageFieldBytes || length > bytes.size() - offset) {
    throw std::runtime_error("Corrupted message payload");
  }
  std::string value(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                    bytes.begin() + static_cast<std::ptrdiff_t>(offset + length));
  offset += length;
  return value;
}

float bounded_similarity_weight(float cosine) {
  if (!std::isfinite(cosine)) {
    return 0.0f;
  }
  return std::exp(std::clamp(cosine, -1.0f, 1.0f));
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
    if (offset != bytes.size()) {
      throw std::runtime_error("Message payload has trailing bytes");
    }
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
  std::vector<uint8_t> encoded;
  encoded.reserve(kTensorFormatMagic.size() + header_bytes + data_bytes);
  encoded.insert(encoded.end(), kTensorFormatMagic.begin(),
                 kTensorFormatMagic.end());
  append_u32(encoded, rank);
  for (int dim : host_state.shape.dims) {
    if (dim <= 0) throw std::invalid_argument("Memory tensor dimensions must be positive");
    append_u32(encoded, static_cast<uint32_t>(dim));
  }
  const float* data = host_state.data();
  for (int64_t index = 0; index < host_state.size; ++index) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(data[index]));
    std::memcpy(&bits, &data[index], sizeof(bits));
    append_u32(encoded, bits);
  }
  return encoded;
}

Tensor deserialize_tensor_state(const std::vector<uint8_t>& encoded) {
  if (encoded.size() < sizeof(uint32_t)) {
    throw std::runtime_error("Corrupted episodic tensor payload");
  }
  size_t offset = 0;
  if (encoded.size() >= kTensorFormatMagic.size() &&
      std::equal(kTensorFormatMagic.begin(), kTensorFormatMagic.end(),
                 encoded.begin())) {
    offset = kTensorFormatMagic.size();
  }
  const uint32_t rank = read_u32(encoded, offset);
  if (rank == 0 || rank > 8) {
    throw std::runtime_error("Invalid episodic tensor rank");
  }
  std::vector<int> dimensions;
  dimensions.reserve(rank);
  size_t element_count = 1;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    const uint32_t dimension = read_u32(encoded, offset);
    if (dimension == 0 || dimension > 1'048'576 ||
        element_count > 1'048'576 / dimension) {
      throw std::runtime_error("Invalid episodic tensor dimensions");
    }
    element_count *= dimension;
    dimensions.push_back(static_cast<int>(dimension));
  }
  const size_t expected_bytes =
      element_count * sizeof(float);
  if (expected_bytes != encoded.size() - offset) {
    throw std::runtime_error("Corrupted episodic tensor data length");
  }
  Tensor tensor(dimensions, Device::CPU);
  float* output = tensor.data();
  for (size_t index = 0; index < element_count; ++index) {
    const uint32_t bits = read_u32(encoded, offset);
    std::memcpy(&output[index], &bits, sizeof(bits));
    if (!std::isfinite(output[index])) {
      throw std::runtime_error(
          "Episodic tensor payload contains NaN or Inf");
    }
  }
  return tensor;
}

std::vector<float> oxtamem_embedding(const Tensor& state) {
  Tensor host = state.get_device() == Device::GPU ? state.cpu() : state;
  std::vector<float> embedding(kOxtaVectorDimensions, 0.0f);
  const float* values = host.data();
  for (int64_t index = 0; index < host.size; ++index) {
    const size_t bucket =
        static_cast<size_t>(index) % kOxtaVectorDimensions;
    // Feature hashing keeps the persistent index dimension fixed while
    // preserving exact zero-padding for the common <=128-dimensional case.
    const float sign =
        index < static_cast<int64_t>(kOxtaVectorDimensions) ||
                ((static_cast<uint64_t>(index) /
                  kOxtaVectorDimensions) & 1u) == 0u
            ? 1.0f
            : -1.0f;
    embedding[bucket] += values[index] * sign;
  }
  const float norm_squared =
      std::inner_product(embedding.begin(), embedding.end(),
                         embedding.begin(), 0.0f);
  if (!std::isfinite(norm_squared) || norm_squared <= 1e-12f) {
    // The Rust vector index rejects zero-norm vectors. A dedicated sentinel
    // gives zero states deterministic, searchable semantics.
    embedding.back() = 1.0f;
  }
  return embedding;
}

} // namespace

MemorySystem::MemorySystem(int chunk_dim) : chunk_size(chunk_dim) {
  if (chunk_dim <= 0 || chunk_dim > 1'048'576) {
    throw std::invalid_argument("MemorySystem chunk_dim must be in 1..1048576");
  }
}

void MemorySystem::validate_state_shape(const Tensor& state,
                                        const char* operation) const {
  if (state.size != chunk_size || state.shape.size() == 0) {
    throw std::invalid_argument(
        std::string(operation) + " requires exactly " +
        std::to_string(chunk_size) + " finite values");
  }
  Tensor host = state.get_device() == Device::GPU ? state.cpu() : state;
  const float* values = host.data();
  for (int64_t index = 0; index < host.size; ++index) {
    if (!std::isfinite(values[index])) {
      throw std::invalid_argument(std::string(operation) +
                                  " rejects non-finite state values");
    }
  }
}

void MemorySystem::add_cluster(const Tensor &state, const std::string &label) {
  (void)label;
  store_episodic(state);
}

void MemorySystem::enable_causal_store(const std::string& path) {
  auto backend = std::make_unique<CausalMemoryStore>(path);
  std::lock_guard<std::mutex> lock(memory_mutex);
  causal_store = std::move(backend);
  oxtamem_store.reset();
  persistence_error.clear();
}

bool MemorySystem::enable_oxtamem_store(const std::string& library_path,
                                        const std::string& store_path,
                                        uint64_t size_mb) {
  std::lock_guard<std::mutex> lock(memory_mutex);
  auto backend = std::make_unique<OxtaMemFFI>();
  if (!backend->load(library_path) || !backend->open(store_path, size_mb)) {
    persistence_error = backend->last_error();
    return false;
  }
  oxtamem_store = std::move(backend);
  causal_store.reset();
  persistence_error.clear();
  return true;
}

std::string MemorySystem::last_persistence_error() const {
  std::lock_guard<std::mutex> lock(memory_mutex);
  return persistence_error;
}

void MemorySystem::store_episodic(const Tensor &state) {
  validate_state_shape(state, "store_episodic");
  Tensor state_cpu = state.get_device() == Device::GPU ? state.cpu() : state.clone();
  const std::vector<uint8_t> serialized = serialize_tensor_state(state_cpu);
  std::lock_guard<std::mutex> lock(memory_mutex);

  // A single configured persistent backend is the source of truth. Publish to
  // it before mutating the runtime index so a failed write cannot create a
  // memory that disappears after restart.
  if (causal_store) {
    causal_store->append("episodic", serialized);
  } else if (oxtamem_store &&
             !oxtamem_store->write_with_vector(
                 "episodic", serialized, oxtamem_embedding(state_cpu))) {
    throw std::runtime_error(
        "OxtaMem episodic write failed: " +
        oxtamem_store->last_error());
  }

  // Initialize cluster's access time
  auto now = std::chrono::system_clock::now();

  // Bound the aggregate runtime footprint, not only each individual cluster.
  // Without this cap, 4096 individually valid clusters could retain millions
  // of tensors and make checkpoints or long training runs unbounded.
  auto runtime_item_count = [&]() {
    size_t count = 0;
    for (const Cluster& cluster : clusters) {
      count += cluster.items.size();
      count += cluster.compressed_items.size();
    }
    return count;
  };
  while (!clusters.empty() &&
         runtime_item_count() >= kMaxRuntimeItems) {
    const auto oldest = std::min_element(
        clusters.begin(), clusters.end(),
        [](const Cluster& lhs, const Cluster& rhs) {
          return lhs.last_access < rhs.last_access;
        });
    if (oldest == clusters.end()) break;
    clusters.erase(oldest);
  }

  // The centroid math below dereferences host pointers, so the incoming
  // state (which may live on GPU) must be materialized on the host first.
  // Without this, state.data() returns a device pointer that is read on
  // the CPU -> undefined behavior / segfault.  The clones stored in the
  // cluster preserve the original device so retrieval semantics and
  // checkpoint contents are unchanged for CPU callers.
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
  if (best_cluster == -1 || best_dist > kClusterRadiusSquared) {
    if (clusters.size() >= kMaxClusters) {
      const auto oldest = std::min_element(
          clusters.begin(), clusters.end(),
          [](const Cluster& lhs, const Cluster& rhs) {
            return lhs.last_access < rhs.last_access;
          });
      if (oldest != clusters.end()) clusters.erase(oldest);
    }
    // Create new
    Cluster c;
    // Runtime memory is host-resident by design: retaining training states on
    // their source GPU would grow VRAM without bound, while every retrieval and
    // TurboQuant operation is host-side anyway.
    c.centroid = state_cpu.clone();
    c.items.push_back(state_cpu.clone());
      c.last_access = now;
      clusters.push_back(c);
  } else {
    // Add to existing
    auto& items = clusters[best_cluster].items;
    if (items.size() >= kMaxItemsPerCluster) {
      items.erase(items.begin());
    }
    items.push_back(state_cpu.clone());
    clusters[best_cluster].last_access = now;
    // Update centroid (Moving Average).  Operate on a host copy then write
    // the result back onto the centroid in its original device, keeping the
    // host-pointer arithmetic safe for GPU-resident centroids.
    float alpha = 0.1f;
    Tensor centroid_cpu = clusters[best_cluster].centroid;
    if (centroid_cpu.size == state_cpu.size) {
      float *c = centroid_cpu.data();
      const float *s = state_cpu.data();
      for (int k = 0; k < state_cpu.size; ++k)
        c[k] = (1 - alpha) * c[k] + alpha * s[k];
      clusters[best_cluster].centroid = centroid_cpu;
    }
  }
}

void MemorySystem::add_instruction(const std::string &instr) {
  if (instr.empty() || instr.size() > kMaxInstructionBytes) {
    throw std::invalid_argument("Instruction must be 1 byte..1 MiB");
  }
  std::lock_guard<std::mutex> lock(memory_mutex);
  if (instructional_memory.size() >= kMaxInstructions) {
    instructional_memory.erase(instructional_memory.begin());
  }
  instructional_memory.push_back(instr);
}

std::vector<Tensor> MemorySystem::retrieval_candidates_locked(
    const Tensor& query_cpu, size_t top_k) {
  if (top_k == 0) return {};
  constexpr size_t max_candidate_bytes = 64ull * 1024ull * 1024ull;
  const size_t state_bytes = static_cast<size_t>(chunk_size) * sizeof(float);
  if (top_k > max_candidate_bytes / state_bytes) {
    throw std::length_error("Requested memory results exceed 64 MiB");
  }
  struct Candidate {
    double score;
    uint64_t hash;
    Tensor state;
    mutable Cluster* owner;
  };
  auto better = [state_bytes](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.hash != b.hash) return a.hash < b.hash;
    return std::memcmp(a.state.data(), b.state.data(), state_bytes) < 0;
  };
  // Keep only the best K unique tensors: retrieval scratch does not grow
  // with the number of clusters or the durable history being considered.
  std::set<Candidate, decltype(better)> candidates(better);
  double query_norm = 0;
  for (int64_t i = 0; i < query_cpu.size; ++i) {
    query_norm += double(query_cpu.data()[i]) * query_cpu.data()[i];
  }
  auto add = [&](const Tensor& state, Cluster* owner = nullptr) {
    if (state.size != query_cpu.size) return;
    Tensor host = state.get_device() == Device::GPU ? state.cpu() : state;
    uint64_t hash = 14695981039346656037ull;
    double dot = 0, norm = 0;
    for (int64_t i = 0; i < host.size; ++i) {
      const float value = host.data()[i];
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite retrieved memory");
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      hash = (hash ^ bits) * 1099511628211ull;
      dot += double(query_cpu.data()[i]) * value;
      norm += double(value) * value;
    }
    const double score = dot / std::sqrt(std::max(query_norm * norm, 1e-16));
    Candidate candidate{score, hash, host.reshape(query_cpu.shape.dims), owner};
    const auto duplicate = candidates.find(candidate);
    if (duplicate != candidates.end()) {
      if (owner) duplicate->owner = owner;
      return;
    }
    if (candidates.size() == top_k) {
      const auto worst = std::prev(candidates.end());
      if (!better(candidate, *worst)) return;
      candidates.erase(worst);
    }
    candidates.insert(std::move(candidate));
  };

  // Durable candidates are always consulted, including after the first new
  // write following a restart. Runtime entries supplement rather than mask it.
  if (oxtamem_store) {
    const size_t shortlist = std::min(max_candidate_bytes / (state_bytes + 40),
        std::min<size_t>(4096, std::max<size_t>(8, top_k * 4)));
    for (const auto& payload : oxtamem_store->search_similar(
             oxtamem_embedding(query_cpu), shortlist)) {
      add(deserialize_tensor_state(payload));
    }
  } else if (causal_store) {
    const size_t depth = std::min<size_t>(1024,
        max_candidate_bytes / (state_bytes + 40));
    for (const auto& payload : causal_store->read_history("episodic", depth)) {
      add(deserialize_tensor_state(payload));
    }
  }

  for (auto& cluster : clusters) {
    for (const auto& state : cluster.items) add(state, &cluster);
    if (tq_engine) {
      for (const auto& packed : cluster.compressed_items) {
        const auto values = tq_engine->decode(packed);
        if (values.size() != static_cast<size_t>(chunk_size)) continue;
        Tensor state = Tensor::uninitialized(query_cpu.shape.dims, Device::CPU);
        std::memcpy(state.data(), values.data(), state_bytes);
        add(state, &cluster);
      }
    }
  }
  // Re-rank with the original full-dimensional tensors, not the 128-D index
  // projection. Stable content ties give the same order after reopening.
  std::vector<Tensor> result;
  result.reserve(candidates.size());
  const auto accessed = std::chrono::system_clock::now();
  for (const auto& candidate : candidates) {
    if (candidate.owner) candidate.owner->last_access = accessed;
    result.push_back(candidate.state);
  }
  return result;
}

Tensor MemorySystem::retrieve(const Tensor& query) {
  Tensor host = query.get_device() == Device::GPU ? query.cpu() : query;
  validate_state_shape(host, "retrieve");
  std::lock_guard<std::mutex> lock(memory_mutex);
  const auto memories = retrieval_candidates_locked(host, 8);
  Tensor context = Tensor::zeros(host.shape.dims, Device::CPU);
  double query_norm = 0;
  for (int64_t i = 0; i < host.size; ++i) query_norm += double(host.data()[i]) * host.data()[i];
  std::vector<double> accumulated(static_cast<size_t>(host.size), 0.0);
  double total = 0;
  for (const auto& state : memories) {
    double dot = 0, norm = 0;
    for (int64_t i = 0; i < host.size; ++i) {
      dot += double(host.data()[i]) * state.data()[i];
      norm += double(state.data()[i]) * state.data()[i];
    }
    const float cosine = static_cast<float>(dot / std::sqrt(std::max(query_norm * norm, 1e-16)));
    const double weight = bounded_similarity_weight(cosine);
    for (int64_t i = 0; i < host.size; ++i) accumulated[static_cast<size_t>(i)] += state.data()[i] * weight;
    total += weight;
  }
  if (total > 0) {
    for (int64_t i = 0; i < host.size; ++i) context.data()[i] = static_cast<float>(accumulated[static_cast<size_t>(i)] / total);
  }
  return query.get_device() == Device::GPU ? context.to(Device::GPU) : context;
}

std::vector<Tensor> MemorySystem::retrieve(const Tensor& query, size_t top_k) {
  if (top_k > 4096) throw std::invalid_argument("retrieve top_k exceeds 4096");
  Tensor host = query.get_device() == Device::GPU ? query.cpu() : query;
  validate_state_shape(host, "retrieve(top_k)");
  std::lock_guard<std::mutex> lock(memory_mutex);
  auto result = retrieval_candidates_locked(host, top_k);
  for (auto& state : result) {
    state = query.get_device() == Device::GPU ? state.to(Device::GPU) : state.clone();
  }
  return result;
}

void MemorySystem::run_auto_dream() {
  std::lock_guard<std::mutex> lock(memory_mutex);
  if (clusters.empty()) return;

  // Initialize engine precisely
  if (!tq_engine && !clusters[0].items.empty()) {
      int dim = chunk_size;
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
              if (cluster.compressed_items.size() > kMaxItemsPerCluster) {
                  const size_t excess =
                      cluster.compressed_items.size() - kMaxItemsPerCluster;
                  cluster.compressed_items.erase(
                      cluster.compressed_items.begin(),
                      cluster.compressed_items.begin() +
                          static_cast<std::ptrdiff_t>(excess));
              }
              cluster.items.clear();
          }
      }
  }
}

void MemorySystem::add_message(const Message &msg) {
    if (msg.role.empty()) {
        throw std::invalid_argument("Message role must not be empty");
    }
    const std::vector<uint8_t> serialized = serialize_message(msg);
    if (serialized.size() > 16ull * 1024ull * 1024ull) {
        throw std::invalid_argument("Serialized message exceeds 16 MiB");
    }
    std::lock_guard<std::mutex> lock(memory_mutex);
    if (causal_store) {
        causal_store->append("messages", serialized);
    } else if (oxtamem_store && !oxtamem_store->write("messages", serialized)) {
        throw std::runtime_error(
            "OxtaMem message write failed: " +
            oxtamem_store->last_error());
    }
    conversation_history.push_back(msg);
    microcompact_messages_locked();
}

std::vector<Message> MemorySystem::recall_recent_messages(size_t depth) const {
    if (depth > 1024) {
        throw std::invalid_argument("Message recall depth exceeds 1024");
    }
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
    for (size_t index = conversation_history.size(); index > begin; --index) {
        messages.push_back(conversation_history[index - 1]);
    }
    return messages;
}

void MemorySystem::microcompact_messages() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    microcompact_messages_locked();
}

void MemorySystem::microcompact_messages_locked() {
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
                    auto decoded = tq_engine->decode(comp);
                    if (decoded.size() == static_cast<size_t>(chunk_size) &&
                        std::all_of(decoded.begin(), decoded.end(),
                                    [](float value) { return std::isfinite(value); })) {
                        decoded_items.push_back(std::move(decoded));
                    }
                } catch (const std::exception& error) {
                    std::cerr << "[MemorySystem] UltraCompact decode failed: "
                              << error.what() << "\n";
                }
            }
            
            if (decoded_items.empty()) continue;

            // 2. Compute Weighted Average (Centroid Rollup)
            const size_t dim = static_cast<size_t>(chunk_size);
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
                Tensor centroid_cpu = cluster.centroid.get_device() == Device::GPU
                                          ? cluster.centroid.cpu()
                                          : cluster.centroid;
                if (centroid_cpu.size == static_cast<int64_t>(dim)) {
                    float* c_data = centroid_cpu.data();
                    for (size_t i = 0; i < dim; ++i) c_data[i] = rollup[i];
                    cluster.centroid = std::move(centroid_cpu);
                }
                
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

std::vector<MemorySystem::Cluster>
MemorySystem::snapshot_runtime_clusters() const {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<Cluster> snapshot;
    snapshot.reserve(clusters.size());
    for (const Cluster& source : clusters) {
        Cluster copy;
        copy.centroid =
            (source.centroid.get_device() == Device::GPU
                 ? source.centroid.cpu()
                 : source.centroid)
                .clone();
        copy.items.reserve(source.items.size());
        for (const Tensor& item : source.items) {
            copy.items.push_back(
                (item.get_device() == Device::GPU ? item.cpu() : item)
                    .clone());
        }
        copy.compressed_items = source.compressed_items;
        copy.last_access = source.last_access;
        snapshot.push_back(std::move(copy));
    }
    return snapshot;
}

void MemorySystem::restore_runtime_clusters(
    const std::vector<Cluster>& snapshot) {
    if (snapshot.size() > kMaxClusters) {
        throw std::invalid_argument(
            "Memory snapshot exceeds cluster limit");
    }
    std::vector<Cluster> staged;
    staged.reserve(snapshot.size());
    bool has_compressed = false;
    size_t compressed_bytes = 0;
    size_t aggregate_items = 0;
    for (const Cluster& source : snapshot) {
        validate_state_shape(source.centroid, "restore centroid");
        if (source.items.size() > kMaxItemsPerCluster ||
            source.compressed_items.size() > kMaxItemsPerCluster) {
            throw std::invalid_argument(
                "Memory snapshot exceeds per-cluster item limit");
        }
        if (source.items.size() >
                kMaxRuntimeItems - aggregate_items) {
            throw std::invalid_argument(
                "Memory snapshot exceeds aggregate item limit");
        }
        aggregate_items += source.items.size();
        if (source.compressed_items.size() >
                kMaxRuntimeItems - aggregate_items) {
            throw std::invalid_argument(
                "Memory snapshot exceeds aggregate item limit");
        }
        aggregate_items += source.compressed_items.size();
        Cluster copy;
        copy.centroid =
            (source.centroid.get_device() == Device::GPU
                 ? source.centroid.cpu()
                 : source.centroid)
                .clone();
        copy.items.reserve(source.items.size());
        for (const Tensor& item : source.items) {
            validate_state_shape(item, "restore item");
            copy.items.push_back(
                (item.get_device() == Device::GPU ? item.cpu() : item)
                    .clone());
        }
        for (const auto& encoded : source.compressed_items) {
            if (encoded.size() > 16ull * 1024ull * 1024ull ||
                compressed_bytes >
                    64ull * 1024ull * 1024ull - encoded.size()) {
                throw std::invalid_argument(
                    "Memory snapshot compressed payload exceeds limit");
            }
            compressed_bytes += encoded.size();
        }
        copy.compressed_items = source.compressed_items;
        has_compressed =
            has_compressed || !copy.compressed_items.empty();
        copy.last_access = source.last_access;
        staged.push_back(std::move(copy));
    }

    std::unique_ptr<tq::TurboQuantEngine> staged_quantizer;
    if (has_compressed) {
        staged_quantizer =
            std::make_unique<tq::TurboQuantEngine>(chunk_size);
        for (const Cluster& cluster : staged) {
            for (const auto& encoded : cluster.compressed_items) {
                const auto decoded = staged_quantizer->decode(encoded);
                if (decoded.size() !=
                        static_cast<size_t>(chunk_size) ||
                    !std::all_of(
                        decoded.begin(), decoded.end(),
                        [](float value) {
                            return std::isfinite(value);
                        })) {
                    throw std::invalid_argument(
                        "Memory snapshot contains an invalid "
                        "compressed item");
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(memory_mutex);
    clusters = std::move(staged);
    tq_engine = std::move(staged_quantizer);
}

} // namespace nsos
