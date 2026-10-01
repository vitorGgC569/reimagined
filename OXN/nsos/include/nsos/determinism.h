#pragma once
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>


namespace nsos {
namespace determinism {

/**
 * Hierarquia de seeds para reprodutibilidade total:
 *
 * Level 0: Global Seed (afeta tudo)
 * Level 1: Component Seed (Tensor, Mamba, TTT, etc.)
 * Level 2: Operation Seed (matmul, conv, etc.)
 * Level 3: Thread Seed (OpenMP/CUDA threads)
 */
class DeterminismManager {
public:
  static DeterminismManager &instance();

  // Level 0: Global
  void set_global_seed(uint64_t seed);
  uint64_t get_global_seed() const;
  uint64_t get_seed_version() const;

  // Level 1: Component
  void set_component_seed(const std::string &component, uint64_t seed);
  uint64_t get_component_seed(const std::string &component) const;

  // Level 2: Operation
  std::mt19937_64 get_rng_for_operation(const std::string &component,
                                        const std::string &operation,
                                        uint64_t sequence_id = 0);

  // Verificação
  bool verify_determinism_context() const;
  std::string get_determinism_report() const;

  // Reset all settings
  void reset();

private:
  DeterminismManager() = default;
  ~DeterminismManager() = default;
  DeterminismManager(const DeterminismManager &) = delete;
  DeterminismManager &operator=(const DeterminismManager &) = delete;

  uint64_t global_seed_ = 0;
  uint64_t seed_version_ = 0;
  std::unordered_map<std::string, uint64_t> component_seeds_;
  mutable std::mutex mutex_;
};

// Opt-in bit-reproducible training (NSOS_DETERMINISTIC=1).  Several GPU
// fast-paths accumulate via order-nondeterministic atomicAdd — the embedding
// gradient scatter, the Mamba grad_A reduction, and the MoE scatter — and the
// loss reduction order can vary on device.  When this returns true, those ops
// are routed to their deterministic host/ordered implementations so two runs
// with the same seed produce byte-identical gradients (trading GPU throughput
// for reproducibility, which checkpoint replay / continual learning require).
// Read once from the env on first call; override at runtime with the setter.
// Default false -> existing (fast, non-deterministic) behavior is unchanged.
bool deterministic_reductions_enabled();
void set_deterministic_reductions(bool enabled);

// Macro para contexto determinístico
#define NSOS_DETERMINISTIC_SCOPE(component, op, seq)                           \
  auto _rng =                                                                  \
      nsos::determinism::DeterminismManager::instance().get_rng_for_operation( \
          component, op, seq)

} // namespace determinism
} // namespace nsos
