#include "inspector.h"
#include <csignal>
#include <cstdlib>
#include <iostream>

#ifdef __linux__
#include <execinfo.h>
#include <unistd.h>
#endif

// Global Signal Handler for Segfaults
void signal_handler(int signal) {
  std::cerr << "\n\n💥 CRITICAL KERNEL PANIC: Signal " << signal
            << " (SIGSEGV/SIGBUS)" << std::endl;
  std::cerr << "   The Engine has encountered a fatal memory error."
            << std::endl;
  std::cerr << "   Dumping Stack Trace:" << std::endl;

#ifdef __linux__
  void *array[20];
  size_t size = backtrace(array, 20);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
#else
  std::cerr << "   [Stack Trace not available on this platform]" << std::endl;
#endif

  std::cerr << "\n   Aborting Process safely..." << std::endl;
  std::exit(signal);
}

#include "nsos/determinism.h"

namespace nsos {
// Install Crash Handler on Init
struct CrashHandlerInstaller {
  CrashHandlerInstaller() {
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
  }
} _installer;

// Definition of global configuration variable
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
} // namespace nsos
