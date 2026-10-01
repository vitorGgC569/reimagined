#ifndef MONITOR_H
#define MONITOR_H

#include "tensor.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nsos {

class Monitor {
public:
  static Monitor &instance() {
    static Monitor inst;
    return inst;
  }

  std::atomic<bool> enabled{false};
  std::atomic<bool> verbose{false};

  void enable() { enabled.store(true, std::memory_order_release); }
  void disable() { enabled.store(false, std::memory_order_release); }
  void set_verbose(bool v) {
    verbose.store(v, std::memory_order_release);
  }

  void check(const Tensor &t, const std::string &tag) {
    if (!enabled.load(std::memory_order_acquire))
      return;
    const bool verbose_now = verbose.load(std::memory_order_acquire);

    // Monitoring is explicitly opt-in, so a synchronous diagnostic copy is
    // preferable to silently skipping GPU tensors. Tensor::to propagates copy
    // and backend failures; a failed health check therefore cannot look clean.
    Tensor inspected =
        t.get_device() == Device::GPU ? t.to(Device::CPU) : t;
    const float *d = inspected.data();
    std::int64_t nan_cnt = 0;
    std::int64_t inf_cnt = 0;
    float max_val = 0.0f;
    double sum = 0.0;

    for (std::int64_t i = 0; i < inspected.size; ++i) {
      float val = d[i];
      if (std::isnan(val))
        nan_cnt++;
      else if (std::isinf(val))
        inf_cnt++;
      else {
        if (std::abs(val) > max_val)
          max_val = std::abs(val);
        if (verbose_now)
          sum += val;
      }
    }

    if (nan_cnt > 0 || inf_cnt > 0) {
      std::cerr << "[MONITOR] 🚨 HEALTH ALERT at " << tag << ": " << nan_cnt
                << " NaNs, " << inf_cnt << " Infs. MaxVal=" << max_val
                << std::endl;
      throw std::runtime_error(
          "Monitor detected non-finite values in tensor '" + tag + "'");
    } else {
      if (verbose_now) {
        const double mean = inspected.size == 0
                                ? 0.0
                                : sum / static_cast<double>(inspected.size);
        std::cout << "[MONITOR] " << tag << " OK. Max=" << max_val
                  << " Mean=" << mean << std::endl;
      }
    }
  }
};

} // namespace nsos

#endif
