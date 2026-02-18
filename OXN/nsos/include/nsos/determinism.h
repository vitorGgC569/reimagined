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
  std::unordered_map<std::string, uint64_t> component_seeds_;
  mutable std::mutex mutex_;
};

// Macro para contexto determinístico
#define NSOS_DETERMINISTIC_SCOPE(component, op, seq)                           \
  auto _rng =                                                                  \
      nsos::determinism::DeterminismManager::instance().get_rng_for_operation( \
          component, op, seq)

} // namespace determinism
} // namespace nsos
