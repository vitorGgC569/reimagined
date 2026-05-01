#include "../include/components.h"
#include "../include/jamba.h"
#include "../include/tokenizer.h"
#include <iostream>
#include <string>
#include <vector>

using namespace nsos;

int main() {
  std::cout << "Initializing NSOS Mini Chat (Jamba-MCTS-BitNet)..."
            << std::endl;

  // 1. Init Tokenizer
  Tokenizer tokenizer;
  int vocab_size = 128; // Pad vocab for embedding

  // 2. Init Model
  // 16 layers, d_model=64 (Mini)
  std::cout << "Loading Jamba Model (16 layers, d_model=64)..." << std::endl;
  JambaModel model(16, 64);

  // Embedding layer (Simple BitLinear projection)
  BitLinear embedding(vocab_size, 64);
  BitLinear head(64, vocab_size);

  // 3. Init System 2 (MCTS)
  Tensor initial_state = Tensor::zeros({1, 64});
  MCTS mcts(initial_state);

  std::cout << "Ready! Type 'exit' to quit." << std::endl;

  std::string input;
  while (true) {
    std::cout << "\nUser: ";
    std::getline(std::cin, input);
    if (input == "exit")
      break;

    std::vector<int> input_ids = tokenizer.encode(input);
    if (input_ids.empty())
      continue;

    // Convert to Tensor (One-hot simulation or direct embed)
    // Here we just project simple embeddings
    // Need [L, D]
    Tensor input_emb = Tensor::zeros({(int)input_ids.size(), 64});
    for (int i = 0; i < input_ids.size(); ++i) {
      // Simple random embedding simulation (since untrained)
      // In real app, we look up row.
      // Here we just fill with seed based on id for determinism
      std::srand(input_ids[i]);
      for (int d = 0; d < 64; ++d) {
        input_emb.at({i, d}) = (float)(std::rand() % 100) / 100.0f;
      }
    }

    std::cout << "NSOS: ";

    // Generate 10 tokens
    Tensor current_emb = input_emb;

    for (int step = 0; step < 10; ++step) {
      // Forward Pass
      Tensor hidden = model.forward(current_emb);

      // Get last token hidden state
      // Slice last row
      Tensor last_h({1, 64});
      int L = hidden.shape[0];
      for (int d = 0; d < 64; ++d) {
        last_h.at({0, d}) = hidden.get({L - 1, d});
      }

      // Logits
      Tensor logits = head.forward(last_h);

      // System 2: MCTS Deciding next token
      // For Chat simulation, we let MCTS explore around argmax
      // Reset MCTS root for new step
      mcts.root = std::make_unique<MCTSNode>(last_h);

      // Run search
      // The MCTS evaluation currently prefers token 5.
      // In a real system, MCTS evaluates states based on Model's Value Head.
      mcts.search(20);

      int next_token = mcts.get_best_action();
      if (next_token == -1)
        next_token = 0; // Fallback

      // Map MCTS dummy tokens (0,1,5) to chars?
      // Our MCTS is dummy, returns 0,1,5.
      // Let's ensure we pick a valid char from vocab
      // Force next_token to be a valid char id if it isn't
      if (next_token < tokenizer.vocab_size) {
        // ok
      } else {
        next_token = next_token % tokenizer.vocab_size;
      }

      // Print
      std::vector<int> out_id = {next_token};
      std::cout << tokenizer.decode(out_id) << std::flush;

      // Append for next step (Autoregressive)
      // Just simulate by keeping context small or using last output
      // Update current_emb to be this new token's embedding
      std::srand(next_token);
      Tensor next_emb({1, 64});
      for (int d = 0; d < 64; ++d) {
        next_emb.at({0, d}) = (float)(std::rand() % 100) / 100.0f;
      }

      // Append (inefficient concat, just replace for demo loop)
      current_emb = next_emb;
    }
    std::cout << std::endl;
  }

  return 0;
}
