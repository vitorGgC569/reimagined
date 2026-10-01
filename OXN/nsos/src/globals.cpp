#include "inspector.h"
#include "nsos/determinism.h"

#include <csignal>
#include <cstdlib>

#ifndef _WIN32
#include <unistd.h>
#endif

// Fatal-signal handling is deliberately opt-in. A library must not replace
// Python's or its host application's process-wide handlers merely by being
// loaded. The handler itself performs only fixed-buffer writes and signal
// primitives; allocation, iostreams, locks and normal process teardown are
// forbidden in this context.
void signal_handler(int signal_number) {
#ifndef _WIN32
  static constexpr char message[] =
      "NSOS fatal signal; restoring the default handler\n";
  (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
#endif
  (void)std::signal(signal_number, SIG_DFL);
  (void)std::raise(signal_number);
#ifndef _WIN32
  ::_exit(128 + signal_number);
#else
  std::_Exit(128 + signal_number);
#endif
}

void install_crash_handler() {
  (void)std::signal(SIGSEGV, signal_handler);
  (void)std::signal(SIGABRT, signal_handler);
}

namespace nsos {

bool GLOBAL_DETERMINISTIC_MODE = true;

void set_global_seed(int seed) {
  determinism::DeterminismManager::instance().set_global_seed(
      static_cast<uint64_t>(seed));
}

int get_global_seed() {
  return static_cast<int>(
      determinism::DeterminismManager::instance().get_global_seed());
}

std::mt19937_64 get_rng() {
  return determinism::DeterminismManager::instance().get_rng_for_operation(
      "global", "generic", 0);
}

}  // namespace nsos
