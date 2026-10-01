#pragma once
#include "determinism.h"
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nsos {
namespace tests {

/**
 * IsolatedTestContext: Previne interferência entre testes e garante
 * reprodutibilidade.
 */
class IsolatedTestContext {
public:
  explicit IsolatedTestContext(uint64_t seed = 42) : seed_(seed) {
    // Isolar seeds
    determinism::DeterminismManager::instance().set_global_seed(seed_);

    // Criar diretório temporário para o teste
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("nsos_test_" + std::to_string(seed_));
    std::filesystem::create_directories(temp_dir_);
  }

  ~IsolatedTestContext() noexcept {
    // Limpar diretório temporário
    std::error_code cleanup_error;
    std::filesystem::remove_all(temp_dir_, cleanup_error);
    if (cleanup_error) {
      std::cerr << "IsolatedTestContext cleanup failed for " << temp_dir_
                << ": " << cleanup_error.message() << '\n';
    }
    // Resetar determinismo
    determinism::DeterminismManager::instance().reset();
  }

  const std::filesystem::path &temp_dir() const { return temp_dir_; }
  uint64_t seed() const { return seed_; }

  void reset() {
    determinism::DeterminismManager::instance().set_global_seed(seed_);
  }

  void isolate_cuda() {
    throw std::logic_error(
        "IsolatedTestContext::isolate_cuda is unsupported; select and "
        "isolate the GPU explicitly in the GPU test harness");
  }

private:
  uint64_t seed_;
  std::filesystem::path temp_dir_;
};

} // namespace tests
} // namespace nsos
