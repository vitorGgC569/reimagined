#ifndef MCTS_REASONING_H
#define MCTS_REASONING_H

#include "tensor.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace nsos {

// Forward declarations
struct ReasoningNode;
class NodePool;

// Estratégia de expansão abstrata
class ExpansionStrategy {
public:
  virtual ~ExpansionStrategy() = default;
  virtual std::vector<Tensor> expand(const Tensor &state, int num_children,
                                     int depth, std::mt19937 &rng) = 0;
  virtual std::vector<float> compute_priors(int num_children) = 0;
};

// Implementação Cauchy-Gaussiana
class CauchyGaussianExpansion : public ExpansionStrategy {
public:
  CauchyGaussianExpansion(float cauchy_scale = 0.01f,
                          float gaussian_std = 0.01f, int sparsity_period = 5);

  std::vector<Tensor> expand(const Tensor &state, int num_children, int depth,
                             std::mt19937 &rng) override;
  std::vector<float> compute_priors(int num_children) override;

private:
  float cauchy_scale_;
  float gaussian_std_;
  int sparsity_period_;
};

// Nó de raciocínio otimizado para cache
struct alignas(64) ReasoningNode {
  // Estado: armazenamos estado completo (COW via Tensor)
  Tensor state;

  // Estatísticas MCTS
  std::atomic<int> visits{0};
  std::atomic<float> value_sum{0.0f};

  // Estrutura da árvore
  ReasoningNode *parent;
  std::vector<ReasoningNode *> children;

  // Metadados
  float prior;
  int depth;
  uint64_t state_hash; // Para detecção de ciclos

  // Construção/destruição controlada pelo pool
  ReasoningNode() = default;
  void init(const Tensor &s, float p, ReasoningNode *par, int d);

  float mean_value() const {
    int v = visits.load(std::memory_order_relaxed);
    return v > 0 ? value_sum.load(std::memory_order_relaxed) / v : 0.0f;
  }

  bool is_leaf() const { return children.empty(); }
  bool is_fully_expanded() const;
};

// Pool de nós para gerenciamento de memória eficiente
class NodePool {
public:
  explicit NodePool(size_t initial_capacity = 1024);
  ~NodePool();

  ReasoningNode *acquire(const Tensor &state, float prior,
                         ReasoningNode *parent, int depth);
  void release(ReasoningNode *node); // Marca como livre
  void clear();                      // Libera toda memória

  size_t size() const { return nodes_.size(); }
  size_t active() const { return active_count_; }

private:
  struct PoolNode {
    alignas(64) ReasoningNode node;
    bool active = false;
  };

  std::vector<std::unique_ptr<PoolNode>> nodes_;
  std::vector<size_t> free_list_;
  size_t active_count_ = 0;
};

// Configuração do MCTS
struct MCTSConfig {
  int num_simulations = 800;
  int max_depth = 20;
  int num_children_per_expansion = 4;
  float c_puct = 1.5f;
  float c_puct_base = 19652.0f; // AlphaZero scaling
  float c_puct_init = 1.25f;
  size_t max_nodes = 100000;               // Limite de memória
  std::chrono::milliseconds time_limit{0}; // 0 = sem limite
  bool use_noise = true;
  float dirichlet_epsilon = 0.25f;
  float dirichlet_alpha = 0.3f;
};

class MCTSReasoning {
public:
  using Evaluator = std::function<float(const Tensor &)>;

  explicit MCTSReasoning(const Tensor &root_state, Evaluator eval,
                         const MCTSConfig &config = {});
  ~MCTSReasoning();

  MCTSReasoning(const MCTSReasoning &) = delete;
  MCTSReasoning &operator=(const MCTSReasoning &) = delete;

  void search();

  Tensor get_best_state() const;
  std::vector<Tensor> get_best_path() const;
  float get_best_value() const;

  size_t num_nodes() const { return pool_->size(); }
  int root_visits() const { return root_ ? root_->visits.load() : 0; }

  void set_expansion_strategy(std::unique_ptr<ExpansionStrategy> strategy);

private:
  ReasoningNode *select(ReasoningNode *root);
  float expand_and_evaluate(ReasoningNode *node);
  void backpropagate(ReasoningNode *node, float value);

  float uct_score(const ReasoningNode *child,
                  const ReasoningNode *parent) const;
  ReasoningNode *select_child(ReasoningNode *parent);
  float rollout(ReasoningNode *node, int depth);
  bool is_cycle(const ReasoningNode *node, uint64_t state_hash) const;

  MCTSConfig config_;
  Evaluator evaluator_;
  std::unique_ptr<NodePool> pool_;
  ReasoningNode *root_;

  std::mt19937 rng_;
  std::unique_ptr<ExpansionStrategy> expansion_strategy_;

  size_t nodes_created_ = 0;
};

} // namespace nsos

#endif // MCTS_REASONING_H
