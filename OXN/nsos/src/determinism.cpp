#include "nsos/determinism.h"
#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace nsos {
namespace determinism {

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
  // If no component seed is set, derive it from global seed
  std::mt19937_64 master_rng(global_seed_);
  // We use a simple hash of the component name to offset the global seed
  std::hash<std::string> hasher;
  return global_seed_ ^ hasher(component);
}

std::mt19937_64
DeterminismManager::get_rng_for_operation(const std::string &component,
                                          const std::string &operation,
                                          uint64_t sequence_id) {
  uint64_t base_seed = get_component_seed(component);
  std::hash<std::string> hasher;
  uint64_t op_seed = base_seed ^ hasher(operation) ^ sequence_id;
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
