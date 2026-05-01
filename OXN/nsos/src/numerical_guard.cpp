#include "../include/numerical_guard.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nsos {

NaNPolicy NumericalGuard::current_policy = NaNPolicy::WARN;
float NumericalGuard::max_value = 1e6f;
std::atomic<int> NumericalGuard::total_fixes{0};

namespace {

Tensor to_host_if_needed(const Tensor &t) {
  return t.get_device() == Device::GPU ? t.cpu() : t;
}

bool classify_value(float value) {
  return std::isnan(value) || std::isinf(value);
}

float sanitize_scalar(float value, int &fixes) {
  if (std::isnan(value) || std::isinf(value)) {
    ++fixes;
    return 0.0f;
  }

  const float limit = std::abs(NumericalGuard::max_value);
  if (limit > 0.0f && std::abs(value) > limit) {
    ++fixes;
    return std::clamp(value, -limit, limit);
  }

  return value;
}

std::string make_context_message(const char *context) {
  return context ? std::string(context) : std::string("unknown");
}

} // namespace

void NumericalGuard::set_policy(NaNPolicy policy) { current_policy = policy; }

void NumericalGuard::set_max_value(float max) {
  max_value = std::max(std::abs(max), 1.0f);
}

bool NumericalGuard::has_numerical_issues(const Tensor &t) {
  if (t.size == 0) {
    return false;
  }

  Tensor host = to_host_if_needed(t);
  const float *ptr = host.data();
  const float limit = std::abs(max_value);
  for (int i = 0; i < host.size; ++i) {
    const float value = ptr[i];
    if (classify_value(value)) {
      return true;
    }
    if (limit > 0.0f && std::abs(value) > limit) {
      return true;
    }
  }
  return false;
}

Tensor NumericalGuard::sanitize(const Tensor &t, const char *context) {
  if (t.size == 0 || current_policy == NaNPolicy::IGNORE) {
    return t;
  }

  if (current_policy == NaNPolicy::THROW) {
    assert_stable(t, context);
    return t;
  }

  Tensor host = to_host_if_needed(t);
  Tensor sanitized = host.clone();
  float *ptr = sanitized.data();
  int fixes = 0;

  for (int i = 0; i < sanitized.size; ++i) {
    const float original = ptr[i];
    float updated = original;
    switch (current_policy) {
    case NaNPolicy::ZERO:
      if (classify_value(original)) {
        ++fixes;
        updated = 0.0f;
      }
      break;
    case NaNPolicy::CLAMP:
      updated = sanitize_scalar(original, fixes);
      break;
    case NaNPolicy::WARN:
      updated = sanitize_scalar(original, fixes);
      break;
    case NaNPolicy::IGNORE:
    case NaNPolicy::THROW:
      break;
    }
    ptr[i] = updated;
  }

  if (fixes > 0) {
    total_fixes.fetch_add(fixes, std::memory_order_relaxed);
    if (current_policy == NaNPolicy::WARN) {
      std::cerr << "[NumericalGuard] sanitized " << fixes
                << " numerical issue(s) in " << make_context_message(context)
                << "\n";
    }
  }

  return sanitized.get_device() == t.get_device() ? sanitized
                                                  : sanitized.to(t.get_device());
}

int NumericalGuard::get_total_fixes() {
  return total_fixes.load(std::memory_order_relaxed);
}

void NumericalGuard::reset_stats() { total_fixes.store(0, std::memory_order_relaxed); }

void NumericalGuard::assert_stable(const Tensor &t, const char *context) {
  if (!has_numerical_issues(t)) {
    return;
  }

  std::ostringstream error;
  error << "NumericalGuard detected NaN/Inf or out-of-range values in "
        << make_context_message(context);
  throw std::runtime_error(error.str());
}

} // namespace nsos
