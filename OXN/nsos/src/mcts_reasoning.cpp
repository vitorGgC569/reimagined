#include "../include/mcts_reasoning.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>

namespace nsos {

// ============================================================================
// ReasoningNode Implementation
// ============================================================================

void ReasoningNode::init(const Tensor &s, float p, ReasoningNode *par, int d) {
  state = s;
  prior = p;
  parent = par;
  depth = d;
  visits.store(0, std::memory_order_relaxed);
  value_sum.store(0.0f, std::memory_order_relaxed);
  children.clear();

  // Hash simples do estado para detecção de ciclos
  state_hash = 0;
  const float *data = s.data();
  if (data) {
    for (int i = 0; i < std::min(s.size, 100); ++i) { // Sample para eficiência
      state_hash = state_hash * 31 + std::hash<float>{}(data[i]);
    }
  }
}

bool ReasoningNode::is_fully_expanded() const {
  return !children.empty() &&
         std::all_of(children.begin(), children.end(),
                     [](ReasoningNode *c) { return c->visits.load() > 0; });
}

// ============================================================================
// NodePool Implementation
// ============================================================================

NodePool::NodePool(size_t initial_capacity) {
  nodes_.reserve(initial_capacity);
  for (size_t i = 0; i < initial_capacity; ++i) {
    nodes_.push_back(std::make_unique<PoolNode>());
    free_list_.push_back(i);
  }
}

NodePool::~NodePool() = default;

ReasoningNode *NodePool::acquire(const Tensor &state, float prior,
                                 ReasoningNode *parent, int depth) {
  size_t idx;
  if (!free_list_.empty()) {
    idx = free_list_.back();
    free_list_.pop_back();
  } else {
    idx = nodes_.size();
    nodes_.push_back(std::make_unique<PoolNode>());
  }

  nodes_[idx]->active = true;
  nodes_[idx]->node.init(state, prior, parent, depth);
  ++active_count_;

  return &nodes_[idx]->node;
}

void NodePool::release(ReasoningNode *node) {
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (&nodes_[i]->node == node && nodes_[i]->active) {
      nodes_[i]->active = false;
      free_list_.push_back(i);
      --active_count_;
      return;
    }
  }
}

void NodePool::clear() {
  for (auto &pn : nodes_) {
    pn->active = false;
  }
  free_list_.clear();
  for (size_t i = 0; i < nodes_.size(); ++i) {
    free_list_.push_back(i);
  }
  active_count_ = 0;
}

// ============================================================================
// CauchyGaussianExpansion Implementation
// ============================================================================

CauchyGaussianExpansion::CauchyGaussianExpansion(float cauchy_scale,
                                                 float gaussian_std,
                                                 int sparsity_period)
    : cauchy_scale_(cauchy_scale), gaussian_std_(gaussian_std),
      sparsity_period_(sparsity_period) {}

std::vector<Tensor> CauchyGaussianExpansion::expand(const Tensor &state,
                                                    int num_children, int depth,
                                                    std::mt19937 &rng) {
  std::vector<Tensor> children;
  children.reserve(num_children);

  float depth_scale = 1.0f / (1.0f + 0.1f * depth);

  std::cauchy_distribution<float> cauchy(0.0f, cauchy_scale_ * depth_scale);
  std::normal_distribution<float> gaussian(0.0f, gaussian_std_ * depth_scale);

  Tensor cpu_state = (state.get_device() == Device::GPU) ? state.cpu() : state;

  for (int k = 0; k < num_children; ++k) {
    Tensor child_state = cpu_state.clone();
    float *data = child_state.data();

    float child_scale = (k == 0) ? 1.0f : (0.3f / std::sqrt((float)(k + 1)));

    for (int j = 0; j < child_state.size; ++j) {
      float noise = (j % sparsity_period_ == 0) ? cauchy(rng) : gaussian(rng);
      data[j] += noise * child_scale;
    }

    if (state.get_device() == Device::GPU) {
      child_state = child_state.to(Device::GPU);
    }

    children.push_back(std::move(child_state));
  }

  return children;
}

std::vector<float> CauchyGaussianExpansion::compute_priors(int num_children) {
  std::vector<float> priors;
  priors.reserve(num_children);

  float sum = 0.0f;
  for (int k = 1; k <= num_children; ++k) {
    float p = 1.0f / k;
    priors.push_back(p);
    sum += p;
  }

  for (auto &p : priors) {
    p /= sum;
  }

  return priors;
}

// ============================================================================
// MCTSReasoning Implementation
// ============================================================================

MCTSReasoning::MCTSReasoning(const Tensor &root_state, Evaluator eval,
                             const MCTSConfig &config)
    : config_(config), evaluator_(std::move(eval)),
      pool_(std::make_unique<NodePool>(config.max_nodes / 4)),
      rng_(std::chrono::steady_clock::now().time_since_epoch().count()),
      expansion_strategy_(std::make_unique<CauchyGaussianExpansion>()) {

  if (!evaluator_) {
    throw std::invalid_argument("Evaluator cannot be null");
  }

  if (root_state.size == 0) {
    throw std::invalid_argument("Root state cannot be empty");
  }

  root_ = pool_->acquire(root_state, 1.0f, nullptr, 0);
}

MCTSReasoning::~MCTSReasoning() = default;

void MCTSReasoning::search() {
  auto start_time = std::chrono::steady_clock::now();

  for (int sim = 0; sim < config_.num_simulations; ++sim) {
    if (config_.time_limit.count() > 0) {
      auto elapsed = std::chrono::steady_clock::now() - start_time;
      if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed) >=
          config_.time_limit) {
        break;
      }
    }

    if (pool_->size() >= config_.max_nodes &&
        pool_->active() >= config_.max_nodes) {
      break;
    }

    ReasoningNode *node = select(root_);

    float value;
    if (node->is_leaf() && node->visits.load() > 0 &&
        node->depth < config_.max_depth) {
      value = expand_and_evaluate(node);
    } else {
      value = evaluator_(node->state);
    }

    backpropagate(node, value);
  }
}

ReasoningNode *MCTSReasoning::select(ReasoningNode *root) {
  ReasoningNode *current = root;

  while (!current->is_leaf() && current->depth < config_.max_depth) {
    if (!current->is_fully_expanded()) {
      return current;
    }
    current = select_child(current);
  }

  return current;
}

ReasoningNode *MCTSReasoning::select_child(ReasoningNode *parent) {
  ReasoningNode *best_child = nullptr;
  float best_score = -std::numeric_limits<float>::infinity();

  bool use_dirichlet = (parent == root_ && config_.use_noise);

  for (size_t i = 0; i < parent->children.size(); ++i) {
    ReasoningNode *child = parent->children[i];
    float score = uct_score(child, parent);

    if (use_dirichlet) {
      score *= (1.0f - config_.dirichlet_epsilon);
      score += config_.dirichlet_epsilon * child->prior;
    }

    if (score > best_score) {
      best_score = score;
      best_child = child;
    }
  }

  return best_child ? best_child
                    : (parent->children.empty() ? parent : parent->children[0]);
}

float MCTSReasoning::uct_score(const ReasoningNode *child,
                               const ReasoningNode *parent) const {
  float parent_visits = static_cast<float>(parent->visits.load());
  float child_visits = static_cast<float>(child->visits.load());

  float q_value = child->mean_value();

  float c_puct = config_.c_puct_init +
                 std::log((1.0f + parent_visits + config_.c_puct_base) /
                          config_.c_puct_base);

  float u_value =
      c_puct * child->prior * std::sqrt(parent_visits) / (1.0f + child_visits);

  return q_value + u_value;
}

float MCTSReasoning::expand_and_evaluate(ReasoningNode *node) {
  auto child_states = expansion_strategy_->expand(
      node->state, config_.num_children_per_expansion, node->depth, rng_);

  auto priors = expansion_strategy_->compute_priors(child_states.size());

  float best_child_value = -std::numeric_limits<float>::infinity();
  ReasoningNode *best_child = nullptr;

  for (size_t k = 0; k < child_states.size(); ++k) {
    uint64_t hash = 0;
    const float *data = child_states[k].data();
    if (data) {
      for (int i = 0; i < std::min(child_states[k].size, 100); ++i) {
        hash = hash * 31 + std::hash<float>{}(data[i]);
      }
    }

    if (is_cycle(node, hash)) {
      continue;
    }

    auto *child =
        pool_->acquire(child_states[k], priors[k], node, node->depth + 1);
    child->state_hash = hash;
    node->children.push_back(child);
    ++nodes_created_;

    float value = evaluator_(child->state);

    child->visits.store(1, std::memory_order_relaxed);
    child->value_sum.store(value, std::memory_order_relaxed);

    if (value > best_child_value) {
      best_child_value = value;
      best_child = child;
    }
  }

  return best_child ? best_child_value : evaluator_(node->state);
}

void MCTSReasoning::backpropagate(ReasoningNode *node, float value) {
  while (node != nullptr) {
    node->visits.fetch_add(1, std::memory_order_relaxed);

    float old_sum = node->value_sum.load(std::memory_order_relaxed);
    float new_sum = old_sum + value;
    node->value_sum.store(new_sum, std::memory_order_relaxed);

    node = node->parent;
  }
}

float MCTSReasoning::rollout(ReasoningNode *node, int depth) {
  if (depth >= config_.max_depth) {
    return evaluator_(node->state);
  }

  auto random_states = expansion_strategy_->expand(node->state, 1, depth, rng_);

  if (random_states.empty()) {
    return evaluator_(node->state);
  }

  return evaluator_(random_states[0]);
}

bool MCTSReasoning::is_cycle(const ReasoningNode *node,
                             uint64_t state_hash) const {
  const ReasoningNode *current = node;
  while (current != nullptr) {
    if (current->state_hash == state_hash) {
      return true;
    }
    current = current->parent;
  }
  return false;
}

Tensor MCTSReasoning::get_best_state() const {
  if (!root_ || root_->children.empty()) {
    return root_ ? root_->state : Tensor();
  }

  const ReasoningNode *best = nullptr;
  int max_visits = -1;
  float best_value = -std::numeric_limits<float>::infinity();

  for (const auto *child : root_->children) {
    int visits = child->visits.load();
    float value = child->mean_value();

    if (visits > max_visits || (visits == max_visits && value > best_value)) {
      max_visits = visits;
      best_value = value;
      best = child;
    }
  }

  return best ? best->state.clone() : root_->state.clone();
}

std::vector<Tensor> MCTSReasoning::get_best_path() const {
  std::vector<Tensor> path;
  if (!root_)
    return path;

  const ReasoningNode *current = root_;
  path.push_back(current->state.clone());

  while (!current->children.empty()) {
    const ReasoningNode *best = nullptr;
    int max_visits = -1;

    for (const auto *child : current->children) {
      int v = child->visits.load();
      if (v > max_visits) {
        max_visits = v;
        best = child;
      }
    }

    if (!best)
      break;
    path.push_back(best->state.clone());
    current = best;
  }

  return path;
}

float MCTSReasoning::get_best_value() const {
  if (!root_ || root_->children.empty()) {
    return root_ ? root_->mean_value() : 0.0f;
  }

  float best_value = -std::numeric_limits<float>::infinity();
  for (const auto *child : root_->children) {
    best_value = std::max(best_value, child->mean_value());
  }
  return best_value;
}

void MCTSReasoning::set_expansion_strategy(
    std::unique_ptr<ExpansionStrategy> strategy) {
  expansion_strategy_ = std::move(strategy);
}

} // namespace nsos
