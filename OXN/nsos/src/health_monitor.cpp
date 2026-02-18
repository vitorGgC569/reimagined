#include "nsos/inspector.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>


namespace nsos {

Inspector &Inspector::instance() {
  static Inspector instance;
  return instance;
}

void Inspector::set_level(int l) {
  std::lock_guard<std::mutex> lock(mutex_);
  level_ = l;
}

void Inspector::enter_scope(const std::string &name) { enter_block(name); }

void Inspector::exit_scope(const Tensor *result) {
  if (result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level_ >= 1) { // LogLevel::TRAFFIC compatible
      std::cout << indent() << "  * Output: [";
      for (size_t i = 0; i < result->shape.size(); ++i)
        std::cout << result->shape[i]
                  << (i < result->shape.size() - 1 ? "x" : "");
      std::cout << "]";
      if (level_ >= 2) { // LogLevel::HEALTH compatible
        print_stats(*result);
      }
      std::cout << std::endl;
    }
  }
  exit_block(""); // Name not used in new exit_block
}

void Inspector::enter_block(const std::string &name, std::source_location loc) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (level_ > 0) {
    std::cout << indent() << "> " << name << " (at " << loc.file_name() << ":"
              << loc.line() << ")" << std::endl;
  }
  scope_stack_.push(name);
  last_op_ = name;

  CheckPoint cp;
  cp.block_name = name;
  cp.timestamp = std::chrono::steady_clock::now();
  cp.location = loc;
  checkpoints_.push_back(cp);

  depth_++;
}

void Inspector::exit_block(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (depth_ > 0)
    depth_--;
  std::string actual_name =
      scope_stack_.empty() ? "Unknown" : scope_stack_.top();
  if (!scope_stack_.empty())
    scope_stack_.pop();

  if (level_ > 0) {
    std::cout << indent() << "< " << actual_name << std::endl;
  }
}

void Inspector::check_tensor_health(const Tensor &tensor,
                                    const std::string &context) {
  if (level_ < 2)
    return; // Scale to LogLevel::HEALTH

  // Check NaNs/Infs (Simplified for now, expecting Tensor to have these helpers
  // or manual check) For now we use print_stats logic if CPU
  if (tensor.get_device() == Device::CPU) {
    const float *d = tensor.data();
    for (int i = 0; i < tensor.size; ++i) {
      if (std::isnan(d[i]))
        panic("NaN detected in tensor: " + context);
      if (std::isinf(d[i]))
        panic("Inf detected in tensor: " + context);
    }
  }
}

void Inspector::check_gradient_health(const Tensor &gradient,
                                      const std::string &param_name) {
  if (level_ < 2)
    return;

  float norm = 0;
  if (gradient.get_device() == Device::CPU) {
    const float *d = gradient.data();
    for (int i = 0; i < gradient.size; ++i)
      norm += d[i] * d[i];
    norm = std::sqrt(norm);

    if (norm < 1e-7)
      log(Severity::WARNING, "Vanishing gradient for " + param_name);
    if (norm > 1e3)
      log(Severity::WARNING,
          "Exploding gradient for " + param_name + ": " + std::to_string(norm));
  }
}

[[noreturn]] void Inspector::panic(const std::string &message,
                                   std::source_location loc) {
  std::cerr << "\n\n==========================================" << std::endl;
  std::cerr << "🔥 FATAL PANIC: " << message << std::endl;
  std::cerr << "Location: " << loc.file_name() << ":" << loc.line() << " ("
            << loc.function_name() << ")" << std::endl;
  std::cerr << "Last Operation: " << last_op_ << std::endl;
  std::cerr << "Scope Stack:" << std::endl;

  std::lock_guard<std::mutex> lock(mutex_);
  auto temp_stack = scope_stack_;
  while (!temp_stack.empty()) {
    std::cerr << " - " << temp_stack.top() << std::endl;
    temp_stack.pop();
  }
  std::cerr << "==========================================\n" << std::endl;
  std::abort();
}

void Inspector::log(Severity sev, const std::string &msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  const char *sev_str = "DEBUG";
  switch (sev) {
  case Severity::INFO:
    sev_str = "INFO";
    break;
  case Severity::WARNING:
    sev_str = "WARNING";
    break;
  case Severity::ERROR:
    sev_str = "ERROR";
    break;
  case Severity::FATAL:
    sev_str = "FATAL";
    break;
  default:
    break;
  }
  std::cout << "[" << sev_str << "] " << msg << std::endl;
}

void Inspector::print_stats(const Tensor &t) {
  if (t.get_device() == Device::GPU) {
    std::cout << " [GPU Statistics Not Available via CPU Inspector]";
    return;
  }

  const float *d = t.data();
  float min_v = 1e9, max_v = -1e9, sum = 0;
  int nans = 0;

  for (int i = 0; i < t.size; ++i) {
    float v = d[i];
    if (std::isnan(v))
      nans++;
    else {
      if (v < min_v)
        min_v = v;
      if (v > max_v)
        max_v = v;
      sum += v;
    }
  }
  float mean = sum / (t.size + 1e-9f);
  std::cout << " | Min:" << min_v << " Max:" << max_v << " Mean:" << mean
            << " NaNs:" << nans;
}

void Inspector::generate_report(const std::string &filename) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream out(filename);
  out << "NSOS Performance & Health Report\n";
  out << "================================\n";
  for (const auto &cp : checkpoints_) {
    out << cp.block_name << " | " << cp.location.file_name() << ":"
        << cp.location.line() << "\n";
  }
}

} // namespace nsos
