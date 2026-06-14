#include "../include/mcts_reasoning.h"
#include "../include/nsos/determinism.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>

namespace nsos {

namespace {

struct SwarmExpansionJob {
  MCTSReasoning *agent = nullptr;
  ReasoningNode *node = nullptr;
  std::vector<ReasoningNode *> children;
};

uint64_t hash_tensor_state(const Tensor& tensor) {
  constexpr uint64_t kOffset = 1469598103934665603ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t hash = kOffset;
  const float* data = tensor.data();
  if (data != nullptr) {
    const int sample_count = static_cast<int>(std::min<int64_t>(tensor.size, 128));
    for (int i = 0; i < sample_count; ++i) {
      const auto* bytes = reinterpret_cast<const unsigned char*>(&data[i]);
      for (size_t j = 0; j < sizeof(float); ++j) {
        hash ^= static_cast<uint64_t>(bytes[j]);
        hash *= kPrime;
      }
    }
  }
  for (int dim : tensor.shape.dims) {
    uint32_t value = static_cast<uint32_t>(dim);
    for (int shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<uint64_t>((value >> shift) & 0xFFu);
      hash *= kPrime;
    }
  }
  return hash;
}

} // namespace

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
    for (int i = 0; i < std::min<int64_t>(s.size, 100); ++i) { // Sample para eficiência
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
    index_by_node_[&nodes_.back()->node] = i;
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
    index_by_node_[&nodes_.back()->node] = idx;
  }

  nodes_[idx]->active = true;
  nodes_[idx]->node.init(state, prior, parent, depth);
  ++active_count_;

  return &nodes_[idx]->node;
}

void NodePool::release(ReasoningNode *node) {
  auto it = index_by_node_.find(node);
  if (it == index_by_node_.end()) {
    return;
  }
  const size_t idx = it->second;
  if (nodes_[idx]->active) {
    nodes_[idx]->active = false;
    free_list_.push_back(idx);
    --active_count_;
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
  const float anchor_mix = 0.85f;
  const float max_cauchy = std::max(0.05f, 4.0f * cauchy_scale_ * depth_scale);
  const float max_gaussian = std::max(0.03f, 4.0f * gaussian_std_ * depth_scale);

  for (int k = 0; k < num_children; ++k) {
    Tensor child_state = cpu_state.clone();
    float *data = child_state.data();

    float child_scale = (k == 0) ? 0.35f : (0.18f / std::sqrt((float)(k + 1)));

    for (int j = 0; j < child_state.size; ++j) {
      float noise = (j % sparsity_period_ == 0) ? cauchy(rng) : gaussian(rng);
      const float clipped =
          (j % sparsity_period_ == 0)
              ? std::clamp(noise, -max_cauchy, max_cauchy)
              : std::clamp(noise, -max_gaussian, max_gaussian);
      data[j] = data[j] * anchor_mix + (data[j] + clipped * child_scale) * (1.0f - anchor_mix);
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
      rng_(),
      expansion_strategy_(std::make_unique<CauchyGaussianExpansion>()) {

  if (!evaluator_) {
    throw std::invalid_argument("Evaluator cannot be null");
  }

  if (root_state.size == 0) {
    throw std::invalid_argument("Root state cannot be empty");
  }

  const uint64_t seed_sequence =
      hash_tensor_state(root_state) ^
      (static_cast<uint64_t>(std::max(config_.num_simulations, 0)) << 32) ^
      static_cast<uint64_t>(std::max(config_.max_depth, 0));
  auto seeded_rng =
      determinism::DeterminismManager::instance().get_rng_for_operation(
          "mcts_reasoning", "search", seed_sequence);
  rng_.seed(static_cast<uint32_t>(seeded_rng()));

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

Tensor MCTSReasoning::search_ultraplan(int num_agents) {
  if (!root_) return Tensor();
  num_agents = std::max(num_agents, 1);
  if (num_agents == 1) {
    search();
    return get_best_state();
  }

  std::vector<std::unique_ptr<MCTSReasoning>> agents;
  agents.reserve(static_cast<size_t>(num_agents));
  for (int i = 0; i < num_agents; ++i) {
    auto agent = std::make_unique<MCTSReasoning>(root_->state, evaluator_, config_);
    agent->set_batch_evaluator(batch_evaluator_);
    agent->rng_.seed(rng_() + static_cast<uint32_t>(7919 * (i + 1)));
    agents.push_back(std::move(agent));
  }

  const auto start_time = std::chrono::steady_clock::now();
  auto budget_exhausted = [&](const MCTSReasoning& agent) {
    if (config_.time_limit.count() > 0) {
      auto elapsed = std::chrono::steady_clock::now() - start_time;
      if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed) >=
          config_.time_limit) {
        return true;
      }
    }
    return agent.pool_->size() >= agent.config_.max_nodes &&
           agent.pool_->active() >= agent.config_.max_nodes;
  };

  for (int sim = 0; sim < config_.num_simulations; ++sim) {
    std::vector<std::pair<MCTSReasoning *, ReasoningNode *>> direct_jobs;
    std::vector<SwarmExpansionJob> expansion_jobs;
    bool has_work = false;

    for (auto& agent : agents) {
      if (budget_exhausted(*agent)) {
        continue;
      }

      has_work = true;
      ReasoningNode *node = agent->select(agent->root_);
      if (node->is_leaf() && node->visits.load(std::memory_order_relaxed) > 0 &&
          node->depth < agent->config_.max_depth) {
        auto children = agent->expand_children(node);
        if (!children.empty()) {
          expansion_jobs.push_back({agent.get(), node, std::move(children)});
        } else {
          direct_jobs.push_back({agent.get(), node});
        }
      } else {
        direct_jobs.push_back({agent.get(), node});
      }
    }

    if (!has_work) {
      break;
    }

    if (!direct_jobs.empty()) {
      std::vector<Tensor> direct_states;
      direct_states.reserve(direct_jobs.size());
      for (const auto& job : direct_jobs) {
        direct_states.push_back(job.second->state);
      }
      auto values = evaluate_states(direct_states);
      for (size_t index = 0; index < direct_jobs.size(); ++index) {
        direct_jobs[index].first->backpropagate(direct_jobs[index].second, values[index]);
      }
    }

    if (!expansion_jobs.empty()) {
      std::vector<Tensor> child_states;
      size_t total_children = 0;
      for (const auto& job : expansion_jobs) {
        total_children += job.children.size();
      }
      child_states.reserve(total_children);
      for (const auto& job : expansion_jobs) {
        for (ReasoningNode* child : job.children) {
          child_states.push_back(child->state);
        }
      }

      auto values = evaluate_states(child_states);
      size_t value_index = 0;
      for (auto& job : expansion_jobs) {
        float best_child_value = -std::numeric_limits<float>::infinity();
        for (ReasoningNode* child : job.children) {
          const float value = values[value_index++];
          child->visits.store(1, std::memory_order_relaxed);
          child->value_sum.store(value, std::memory_order_relaxed);
          best_child_value = std::max(best_child_value, value);
        }
        job.agent->backpropagate(job.node, best_child_value);
      }
    }
  }

  float global_best_score = -std::numeric_limits<float>::infinity();
  Tensor global_best_state;
  for (const auto& agent : agents) {
    const float value = agent->get_best_value();
    if (value > global_best_score) {
      global_best_score = value;
      global_best_state = agent->get_best_state();
    }
  }

  return global_best_state.size > 0 ? global_best_state : root_->state.clone();
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
  auto evaluable_children = expand_children(node);
  if (evaluable_children.empty()) {
    return evaluator_(node->state);
  }

  std::vector<Tensor> batch_states;
  batch_states.reserve(evaluable_children.size());
  for (ReasoningNode* child : evaluable_children) {
    batch_states.push_back(child->state);
  }

  auto values = evaluate_states(batch_states);
  float best_child_value = -std::numeric_limits<float>::infinity();
  for (size_t idx = 0; idx < evaluable_children.size(); ++idx) {
    ReasoningNode* child = evaluable_children[idx];
    const float value = values[idx];
    child->visits.store(1, std::memory_order_relaxed);
    child->value_sum.store(value, std::memory_order_relaxed);
    best_child_value = std::max(best_child_value, value);
  }

  return best_child_value;
}

void MCTSReasoning::backpropagate(ReasoningNode *node, float value) {
  while (node != nullptr) {
    node->visits.fetch_add(1, std::memory_order_relaxed);
    node->value_sum.fetch_add(value, std::memory_order_relaxed);

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

void MCTSReasoning::set_batch_evaluator(BatchEvaluator evaluator) {
  batch_evaluator_ = std::move(evaluator);
}

std::vector<ReasoningNode *> MCTSReasoning::expand_children(ReasoningNode *node) {
  auto child_states = expansion_strategy_->expand(
      node->state, config_.num_children_per_expansion, node->depth, rng_);
  auto priors = expansion_strategy_->compute_priors(static_cast<int>(child_states.size()));

  std::vector<ReasoningNode *> evaluable_children;
  evaluable_children.reserve(child_states.size());
  for (size_t k = 0; k < child_states.size(); ++k) {
    uint64_t hash = 0;
    const float *data = child_states[k].data();
    if (data) {
      for (int i = 0; i < std::min<int64_t>(child_states[k].size, 100); ++i) {
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
    evaluable_children.push_back(child);
  }

  return evaluable_children;
}

std::vector<float> MCTSReasoning::evaluate_states(const std::vector<Tensor>& states) const {
  std::vector<float> values;
  if (states.empty()) {
    return values;
  }

  if (batch_evaluator_) {
    values = batch_evaluator_(states);
  }

  if (values.size() != states.size()) {
    values.clear();
    values.reserve(states.size());
    for (const Tensor& state : states) {
      values.push_back(evaluator_(state));
    }
  }

  return values;
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
