#include "jamba.h"
#include "tokenizer.h"
#include "trainer.h"
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace nsos;

/**
 * NSOS Standalone Circuit Trainer
 * Entry point for direct C++ training of the Jamba-Hybrid model.
 */
int main() {
  std::cout << "Starting NSOS Standalone Circuit Trainer..." << std::endl;

  // 1. Model Configuration
  int num_layers = 4; // Minimal 'brain' configuration
  int d_model = 256;
  int vocab_size = 256;
  Device device = Device::CPU;

#ifdef USE_CUDA
  device = Device::GPU;
  std::cout << "Using GPU (CUDA) for training." << std::endl;
#else
  std::cout << "Using CPU for training." << std::endl;
#endif

  // 2. Initialize Model
  auto model =
      std::make_unique<JambaModel>(num_layers, d_model, vocab_size, device);
  std::cout << "Model initialized with " << num_layers << " layers and "
            << d_model << " dimensions." << std::endl;

  // 3. Prepare Dummy Dataset for Training
  // In a real scenario, this would load from a file (e.g., wiki.train.raw)
  std::vector<int> tokens;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, vocab_size - 1);

  int num_tokens = 4096;
  for (int i = 0; i < num_tokens; ++i) {
    tokens.push_back(dist(rng));
  }
  std::cout << "Generated dummy dataset with " << num_tokens << " tokens."
            << std::endl;

  // 4. Initialize Trainer
  float learning_rate = 0.001f;
  Trainer trainer(model.get(), learning_rate);

  // 5. Run Training Loop
  int epochs = 1;     // 1 epoch is enough for a heavy benchmark
  int batch_size = 2; // Reduced batch for CPU stability at 256 dim
  int seq_len = 128;

  std::cout << "🚀 Starting training loop (Epochs: " << epochs
            << ", Batch size: " << batch_size << ", Seq len: " << seq_len << ")"
            << std::endl;

  auto callback = [](int step, float loss) {
    std::cout << "[Step " << step << "] Current Loss: " << loss << std::endl;
  };

  try {
    trainer.train_loop(tokens, epochs, batch_size, seq_len, callback);
    std::cout << "\n✅ Training complete! Checkpoint saved as "
                 "circuit_brain_checkpoint.bin"
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "💥 Error during training: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
