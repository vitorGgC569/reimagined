#ifndef NUMERICAL_GUARD_H
#define NUMERICAL_GUARD_H

#include "tensor.h"
#include <atomic>
#include <cmath>
#include <iostream>
#include <string>

namespace nsos {

/**
 * @brief Policy for handling NaN/Inf values
 */
enum class NaNPolicy {
  IGNORE, // Do nothing (unsafe, but fastest)
  WARN,   // Log warning and continue
  CLAMP,  // Clamp values to [-max, max] and replace NaN with 0
  ZERO,   // Replace NaN/Inf with 0
  THROW   // Throw exception (strictest)
};

/**
 * @brief Centralized numerical stability guard
 *
 * Provides unified handling of NaN/Inf values across the entire codebase.
 * This replaces scattered sanitize() calls with a configurable system.
 *
 * Usage:
 *   NumericalGuard::set_policy(NaNPolicy::CLAMP);
 *   Tensor safe = NumericalGuard::sanitize(tensor, "layer_name");
 */
class NumericalGuard {
public:
  // Static members - defined in numerical_guard.cpp
  static NaNPolicy current_policy;
  static float max_value;
  static std::atomic<int> total_fixes;

  /**
   * @brief Set the global NaN handling policy
   */
  static void set_policy(NaNPolicy policy);
  static void set_max_value(float max);
  static bool has_numerical_issues(const Tensor &t);
  static Tensor sanitize(const Tensor &t, const char *context = nullptr);
  static int get_total_fixes();
  static void reset_stats();
  static void assert_stable(const Tensor &t, const char *context = nullptr);
};

// Convenience macro for guarding tensors with automatic context
#define NSOS_GUARD(tensor) nsos::NumericalGuard::sanitize(tensor, __FUNCTION__)

#define NSOS_ASSERT_STABLE(tensor)                                             \
  nsos::NumericalGuard::assert_stable(tensor, __FUNCTION__)

} // namespace nsos

#endif // NUMERICAL_GUARD_H
