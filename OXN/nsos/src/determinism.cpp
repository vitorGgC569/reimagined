#include "nsos/determinism.h"
#include "runtime_execution_identity.h"
#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace nsos {
namespace determinism {

namespace {

// std::hash<std::string> is deliberately implementation-defined, so seeds made
// with it are not portable between libstdc++, libc++ and MSVC.  FNV-1a gives us
// a stable byte-level hash and splitmix64 removes its linear structure before
// the value is used as an RNG seed.
uint64_t stable_string_hash(const std::string &value) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : value) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t splitmix64(uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

uint64_t combine_seed(uint64_t seed, uint64_t value) {
  return splitmix64(seed ^ splitmix64(value));
}

}  // namespace

// -1 = not yet read from env; 0/1 = explicit.  Atomic so it is safe to query
// from any thread on the training hot path.
static std::atomic<int> g_deterministic_reductions{-1};

bool deterministic_reductions_enabled() {
  int v = g_deterministic_reductions.load(std::memory_order_relaxed);
  if (v < 0) {
    const char *e = std::getenv("NSOS_DETERMINISTIC");
    v = (e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 1 : 0;
    g_deterministic_reductions.store(v, std::memory_order_relaxed);
  }
  return v != 0;
}

void set_deterministic_reductions(bool enabled) {
  RuntimeExecutionPolicyMutationGuard mutation;
  g_deterministic_reductions.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

DeterminismManager &DeterminismManager::instance() {
  static DeterminismManager instance;
  return instance;
}

void DeterminismManager::set_global_seed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  global_seed_ = seed;
  ++seed_version_;
  // When global seed is set, we use it to derive seeds for all components by
  // default
  std::mt19937_64 master_rng(global_seed_);
  for (auto &pair : component_seeds_) {
    pair.second = master_rng();
  }
}

uint64_t DeterminismManager::get_global_seed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return global_seed_;
}

uint64_t DeterminismManager::get_seed_version() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return seed_version_;
}

void DeterminismManager::set_component_seed(const std::string &component,
                                            uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  component_seeds_[component] = seed;
  ++seed_version_;
}

uint64_t
DeterminismManager::get_component_seed(const std::string &component) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = component_seeds_.find(component);
  if (it != component_seeds_.end()) {
    return it->second;
  }
  // If no component seed is set, derive it portably from the global seed.
  return combine_seed(global_seed_, stable_string_hash(component));
}

std::mt19937_64
DeterminismManager::get_rng_for_operation(const std::string &component,
                                          const std::string &operation,
                                          uint64_t sequence_id) {
  uint64_t base_seed = get_component_seed(component);
  uint64_t op_seed = combine_seed(base_seed, stable_string_hash(operation));
  op_seed = combine_seed(op_seed, sequence_id);
  return std::mt19937_64(op_seed);
}

bool DeterminismManager::verify_determinism_context() const {
  std::lock_guard<std::mutex> lock(mutex_);
  // Simple heuristic: if global_seed is 0, we might be in non-deterministic
  // mode unless explicitly intended.
  return global_seed_ != 0;
}

std::string DeterminismManager::get_determinism_report() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::stringstream ss;
  ss << "NSOS Determinism Report\n";
  ss << "========================\n";
  ss << "Global Seed: 0x" << std::hex << std::setw(16) << std::setfill('0')
     << global_seed_ << "\n";
  ss << "Component Seeds:\n";
  for (const auto &pair : component_seeds_) {
    ss << "  - " << pair.first << ": 0x" << std::hex << std::setw(16)
       << std::setfill('0') << pair.second << "\n";
  }
  return ss.str();
}

void DeterminismManager::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  global_seed_ = 0;
  component_seeds_.clear();
  ++seed_version_;
}

} // namespace determinism
} // namespace nsos
