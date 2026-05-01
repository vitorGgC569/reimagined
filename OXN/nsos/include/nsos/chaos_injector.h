#pragma once
#include "../tensor.h"
#include <limits>
#include <random>
#include <string>
#include <unordered_map>

namespace nsos {
namespace tests {

/**
 * ChaosInjector: Injeção de falhas para Chaos Engineering
 */
class ChaosInjector {
public:
  enum class FaultType {
    MEMORY_CORRUPTION,
    GPU_TIMEOUT,
    NETWORK_PARTITION,
    NaN_INJECTION,
    RANDOM_SLEEP
  };

  ChaosInjector(uint64_t seed = 0xDEADBEEF) : rng_(seed) {}

  void inject_fault(FaultType type, double probability) {
    fault_probs_[type] = probability;
  }

  bool should_trigger(FaultType type) {
    auto it = fault_probs_.find(type);
    if (it == fault_probs_.end())
      return false;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < it->second;
  }

  void apply_nan_injection(Tensor &t) {
    if (!should_trigger(FaultType::NaN_INJECTION))
      return;

    if (t.get_device() == Device::CPU) {
      float *d = t.data();
      std::uniform_int_distribution<int> dist(0, t.size - 1);
      d[dist(rng_)] = std::numeric_limits<float>::quiet_NaN();
    }
  }

  void reset() { fault_probs_.clear(); }

private:
  std::mt19937_64 rng_;
  std::unordered_map<FaultType, double> fault_probs_;
};

} // namespace tests
} // namespace nsos
