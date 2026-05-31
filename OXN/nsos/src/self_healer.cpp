#include "self_healer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>

namespace nsos {

NeuralSelfHealer::NeuralSelfHealer(const HealingConfig &cfg) : cfg_(cfg) {}

// ============================================================================
// 1. ConfidenceGate
// ============================================================================
std::pair<float, float>
NeuralSelfHealer::measure_confidence(const Tensor &logits) const {
  const int vocab = logits.shape.back();
  const float *raw = logits.data() + (logits.size - vocab);

  float max_val = *std::max_element(raw, raw + vocab);
  float sum_exp = 0.0f;
  std::vector<float> probs(vocab);
  for (int i = 0; i < vocab; ++i) {
    probs[i] = std::exp(raw[i] - max_val);
    sum_exp += probs[i];
  }
  float inv = 1.0f / std::max(sum_exp, 1e-9f);
  float max_prob = 0.0f;
  float entropy = 0.0f;
  for (int i = 0; i < vocab; ++i) {
    probs[i] *= inv;
    if (probs[i] > max_prob)
      max_prob = probs[i];
    if (probs[i] > 1e-9f)
      entropy -= probs[i] * std::log(probs[i]);
  }
  entropy /= std::log(2.0f);
  return {max_prob, entropy};
}

// ============================================================================
// 2. ConsistencyCheck
// ============================================================================
float NeuralSelfHealer::measure_consistency(const std::vector<int> &prompt,
                                            float temperature,
                                            int samples) const {
  if (!decoder_ || samples <= 1)
    return 1.0f;

  std::vector<int> first_tokens;
  first_tokens.reserve(samples);

  for (int s = 0; s < samples; ++s) {
    auto gen =
        decoder_(prompt, 1, temperature, 1.0f, 50, cfg_.eos_token_id, nullptr);
    if (!gen.empty()) {
      first_tokens.push_back(gen.back());
    }
  }

  if (first_tokens.empty())
    return 0.0f;

  std::unordered_map<int, int> freq;
  for (int t : first_tokens)
    freq[t]++;
  int max_count = 0;
  for (auto &[tok, cnt] : freq)
    max_count = std::max(max_count, cnt);
  return static_cast<float>(max_count) /
         static_cast<float>(first_tokens.size());
}

// ============================================================================
// 3. SemanticValidator
// ============================================================================
bool NeuralSelfHealer::validate_structure(
    const std::vector<int> &tokens) const {
  switch (validation_type_) {
  case ValidationType::NONE:
    return true;
  case ValidationType::BALANCED_BRACKETS:
    return check_balanced_brackets(tokens);
  case ValidationType::JSON_STRUCTURE:
    return check_json_structure(tokens);
  case ValidationType::MATH_EXPRESSION:
    return check_balanced_brackets(tokens);
  case ValidationType::CUSTOM:
    return custom_validator_ ? custom_validator_(tokens) : true;
  }
  return true;
}

bool NeuralSelfHealer::check_balanced_brackets(
    const std::vector<int> &tokens) const {
  std::vector<int> stack;
  stack.reserve(tokens.size());

  auto matches = [](int open, int close) {
    return (open == '(' && close == ')') || (open == '[' && close == ']') ||
           (open == '{' && close == '}');
  };

  for (int token : tokens) {
    if (token == '(' || token == '[' || token == '{') {
      stack.push_back(token);
    } else if (token == ')' || token == ']' || token == '}') {
      if (stack.empty() || !matches(stack.back(), token)) {
        return false;
      }
      stack.pop_back();
    }
  }

  return stack.empty();
}

bool NeuralSelfHealer::check_json_structure(
    const std::vector<int> &tokens) const {
  if (tokens.size() < 2 || !check_balanced_brackets(tokens)) {
    return false;
  }

  int brace_depth = 0;
  int bracket_depth = 0;
  bool saw_container = false;

  for (int token : tokens) {
    if (token == '{') {
      ++brace_depth;
      saw_container = true;
    } else if (token == '}') {
      --brace_depth;
      if (brace_depth < 0)
        return false;
    } else if (token == '[') {
      ++bracket_depth;
      saw_container = true;
    } else if (token == ']') {
      --bracket_depth;
      if (bracket_depth < 0)
        return false;
    }
  }

  return saw_container && brace_depth == 0 && bracket_depth == 0;
}

// ============================================================================
// Core: generate_healed
// ============================================================================
std::vector<int> NeuralSelfHealer::generate_healed(
    const std::vector<int> &prompt, int max_tokens, HealingReport &report,
    Context *ctx) {
  if (!decoder_ || !extractor_) {
    report.failure_reason = "decoder ou extractor nao registrado";
    report.success = false;
    return {};
  }

  report = {};
  float temperature = cfg_.temperature_start;

  for (int attempt = 0; attempt < cfg_.max_retries; ++attempt) {
    report.attempts = attempt + 1;

    std::vector<int> candidates =
        decoder_(prompt, max_tokens, temperature, 0.9f, 50,
                 cfg_.eos_token_id, ctx);

    Tensor probe_logits = extractor_(prompt);
    auto [conf, entropy] = measure_confidence(probe_logits);
    report.final_confidence = conf;
    report.final_entropy = entropy;

    bool conf_ok = (conf >= cfg_.confidence_threshold);
    bool entropy_ok = (entropy <= cfg_.entropy_threshold);

    if (!conf_ok || !entropy_ok) {
      std::cout << "[SelfHealer] Tentativa " << attempt + 1
                << ": confianca=" << conf << " entropia=" << entropy
                << " abaixo do threshold, ajustando temperatura.\n";
      temperature *= cfg_.temperature_decay;
      continue;
    }

    float consistency =
        measure_consistency(prompt, temperature, cfg_.consistency_samples);
    report.consistency_ok = (consistency >= 0.5f);

    if (!report.consistency_ok) {
      std::cout << "[SelfHealer] Tentativa " << attempt + 1
                << ": consistencia=" << consistency
                << " divergente, reduzindo temperatura.\n";
      temperature *= cfg_.temperature_decay;
      continue;
    }

    bool struct_ok = validate_structure(candidates);
    if (!struct_ok) {
      std::cout << "[SelfHealer] Tentativa " << attempt + 1
                << ": validacao estrutural falhou, regenerando.\n";
      temperature *= cfg_.temperature_decay;
      continue;
    }

    report.success = true;
    std::cout << "[SelfHealer] Output aprovado na tentativa " << attempt + 1
              << " (conf=" << conf << ", entropy=" << entropy
              << ", consistency=" << consistency << ")\n";
    return candidates;
  }

  report.failure_reason =
      "Esgotou " + std::to_string(cfg_.max_retries) +
      " tentativas sem aprovar todos os gates";
  std::cerr << "[SelfHealer] AVISO: " << report.failure_reason << "\n";
  return decoder_(prompt, max_tokens, temperature, 0.9f, 50,
                  cfg_.eos_token_id, ctx);
}

// ============================================================================
// verify: post-hoc check de sequencia ja gerada
// ============================================================================
bool NeuralSelfHealer::verify(const std::vector<int> &tokens,
                              HealingReport &report) {
  report = {};
  report.attempts = 1;

  if (!extractor_) {
    report.failure_reason = "extractor nao registrado";
    return false;
  }

  Tensor probe_logits = extractor_(tokens);
  auto [conf, entropy] = measure_confidence(probe_logits);
  report.final_confidence = conf;
  report.final_entropy = entropy;

  bool conf_ok = (conf >= cfg_.confidence_threshold);
  bool entropy_ok = (entropy <= cfg_.entropy_threshold);
  bool struct_ok = validate_structure(tokens);

  report.success = conf_ok && entropy_ok && struct_ok;
  if (!report.success) {
    std::ostringstream oss;
    if (!conf_ok)
      oss << "confianca=" << conf << " < " << cfg_.confidence_threshold
          << "; ";
    if (!entropy_ok)
      oss << "entropia=" << entropy << " > " << cfg_.entropy_threshold
          << "; ";
    if (!struct_ok)
      oss << "falha estrutural; ";
    report.failure_reason = oss.str();
  }
  return report.success;
}

std::vector<int> generate_with_self_healing(JambaModel& model,
                                            const std::vector<int>& prompt,
                                            int max_tokens,
                                            HealingReport& report,
                                            const HealingConfig& cfg) {
  NeuralSelfHealer healer(cfg);
  D2FDecoder decoder(&model);
  healer.set_logits_extractor([&model](const std::vector<int>& ids) -> Tensor {
    Context c;
    return model.forward_ids(ids, &c);
  });
  healer.set_token_decoder([&decoder](const std::vector<int>& p, int max_len, float temp,
                                      float top_p, int top_k, int eos, Context* ctx) {
    return decoder.generate(p, max_len, ctx, temp, top_p, top_k, eos);
  });
  return healer.generate_healed(prompt, max_tokens, report);
}

} // namespace nsos
