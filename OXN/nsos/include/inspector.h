#pragma once
#include "tensor.h"
#include <chrono>
#include <iostream>
#include <mutex>
#ifdef __cpp_lib_source_location
#include <source_location>
using SourceLoc = std::source_location;
#define NSOS_CUR_LOC std::source_location::current()
#else
struct SourceLoc {
  const char *file_name() const { return "unknown"; }
  uint32_t line() const { return 0; }
  static SourceLoc current() { return {}; }
};
#define NSOS_CUR_LOC SourceLoc::current()
#endif
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace nsos {

/**
 * Sistema de observabilidade e prevenção de erros - Versão Industrial
 */
class Inspector {
public:
  enum class Severity { DEBUG, INFO, WARNING, ERROR, FATAL };

  struct CheckPoint {
    std::string block_name;
    std::chrono::steady_clock::time_point timestamp;
    SourceLoc location;
    std::unordered_map<std::string, double> metrics;
  };

  static Inspector &instance();

  // Compatibility with old API
  void set_level(int l);
  void enter_scope(const std::string &name);
  void exit_scope(const Tensor *result = nullptr);

  // New API
  void enter_block(const std::string &name, SourceLoc loc = NSOS_CUR_LOC);
  void exit_block(const std::string &name);

  // Health monitoring
  void check_tensor_health(const Tensor &tensor, const std::string &context);
  void check_gradient_health(const Tensor &gradient,
                             const std::string &param_name);

  // Panic handling
  [[noreturn]] void panic(const std::string &message = "General Failure",
                          SourceLoc loc = NSOS_CUR_LOC);

  // Relatórios
  void generate_report(const std::string &filename) const;

  void log(Severity sev, const std::string &msg);

private:
  Inspector() = default;
  ~Inspector() = default;
  Inspector(const Inspector &) = delete;
  Inspector &operator=(const Inspector &) = delete;

  int level_ = 0;
  int depth_ = 0;
  std::stack<std::string> scope_stack_;
  std::vector<CheckPoint> checkpoints_;
  std::unordered_map<std::string, std::vector<double>> metrics_history_;
  mutable std::mutex mutex_;
  std::string last_op_ = "None";

  std::string indent() { return std::string(depth_ * 2, ' '); }

  void print_stats(const Tensor &t);
};

// Macros para fácil uso
#define NSOS_INSPECT_ENTER(name) nsos::Inspector::instance().enter_block(name)

#define NSOS_INSPECT_EXIT(name) nsos::Inspector::instance().exit_block(name)

#define NSOS_CHECK_TENSOR(tensor, ctx)                                         \
  nsos::Inspector::instance().check_tensor_health(tensor, ctx)

} // namespace nsos

// Global signal handler declarations
void signal_handler(int signum);
// Explicit opt-in. NSOS never takes ownership of process-wide fatal signals
// merely because the library was loaded.
void install_crash_handler();
