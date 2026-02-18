#pragma once
#include "determinism.h"
#include <cstdint>
#include <filesystem>
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

  ~IsolatedTestContext() {
    // Limpar diretório temporário
    std::filesystem::remove_all(temp_dir_);
    // Resetar determinismo
    determinism::DeterminismManager::instance().reset();
  }

  const std::filesystem::path &temp_dir() const { return temp_dir_; }
  uint64_t seed() const { return seed_; }

  void reset() {
    determinism::DeterminismManager::instance().set_global_seed(seed_);
  }

  // Placeholder para isolamento de hardware específico (CUDA, MPI)
  void isolate_cuda() {
    // Em uma implementação real, poderíamos selecionar um dispositivo
    // específico ou resetar o estado da GPU aqui.
  }

private:
  uint64_t seed_;
  std::filesystem::path temp_dir_;
};

} // namespace tests
} // namespace nsos
