#ifndef MONITOR_H
#define MONITOR_H

#include "tensor.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace nsos {

class Monitor {
public:
  static Monitor &instance() {
    static Monitor inst;
    return inst;
  }

  bool enabled = false;
  bool verbose = false;

  void enable() { enabled = true; }
  void disable() { enabled = false; }
  void set_verbose(bool v) { verbose = v; }

  void check(const Tensor &t, const std::string &tag) {
    if (!enabled)
      return;

    // Use sanitize logic but just logging
    // Note: Accessing data requires device check to avoid crash
    if (t.get_device() == Device::GPU) {
      // Todo: GPU Kernel for IsNan/IsInf
      if (verbose) {
        // std::cout << "[MONITOR] " << tag << " (GPU - Check Skipped)" <<
        // std::endl;
      }
      return;
    }

    const float *d = t.data();
    int nan_cnt = 0;
    int inf_cnt = 0;
    float max_val = 0.0f;
    float sum = 0.0f;

    for (int i = 0; i < t.size; ++i) {
      float val = d[i];
      if (std::isnan(val))
        nan_cnt++;
      else if (std::isinf(val))
        inf_cnt++;
      else {
        if (std::abs(val) > max_val)
          max_val = std::abs(val);
        if (verbose)
          sum += val;
      }
    }

    if (nan_cnt > 0 || inf_cnt > 0) {
      std::cerr << "[MONITOR] 🚨 HEALTH ALERT at " << tag << ": " << nan_cnt
                << " NaNs, " << inf_cnt << " Infs. MaxVal=" << max_val
                << std::endl;
      // throw std::runtime_error("Numerical instability detected"); // Optional
      // strict mode
    } else {
      if (verbose) {
        float mean = sum / (t.size + 1e-6);
        std::cout << "[MONITOR] " << tag << " OK. Max=" << max_val
                  << " Mean=" << mean << std::endl;
      }
    }
  }
};

} // namespace nsos

#endif
