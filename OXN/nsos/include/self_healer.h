#pragma once

#include "tensor.h"
#include "jamba.h"
#include <functional>
#include <string>
#include <vector>
#include <cmath>

namespace nsos {

// ============================================================================
// NeuralSelfHealer — Sistema de Auto-Correção sem dependência externa
// ============================================================================
// Implementa verificação e correção de outputs do modelo via 3 mecanismos:
//   1. ConfidenceGate  — rejeita geração de baixa confiança (entropia alta)
//   2. ConsistencyCheck — detecta inconsistência entre múltiplas amostras
//   3. SemanticValidator — valida estrutura sintática do output (JSON, math)
// ============================================================================

struct HealingConfig {
    float confidence_threshold = 0.5f;  // Min prob do token mais provável
    float entropy_threshold    = 3.5f;  // Max entropia aceitável (bits)
    int   consistency_samples  = 3;     // Número de amostras para consenso
    int   max_retries          = 4;     // Tentativas máximas de correção
    float temperature_start    = 0.7f;  // Temperatura inicial
    float temperature_decay    = 0.85f; // Fator de decay por tentativa
    int   eos_token_id         = 0;     // Token de fim de sequência
};

enum class ValidationType {
    NONE,         // Sem validação semântica
    BALANCED_BRACKETS, // Verifica balanceamento de () [] {}
    JSON_STRUCTURE,    // Verifica abertura/fechamento de JSON
    MATH_EXPRESSION,   // Verifica expressões matemáticas balanceadas
    CUSTOM             // Função de validação externa
};

struct HealingReport {
    bool   success         = false;
    int    attempts        = 0;
    float  final_confidence = 0.0f;
    float  final_entropy    = 0.0f;
    bool   consistency_ok   = false;
    std::string failure_reason;
};

class NeuralSelfHealer {
public:
    using TokenDecoder  = std::function<std::vector<int>(
        const std::vector<int>& prompt, int max_len, float temperature,
        float top_p, int top_k, int eos_token, Context* ctx)>;

    using LogitsExtractor = std::function<Tensor(
        const std::vector<int>& ids)>;

    using CustomValidator = std::function<bool(const std::vector<int>& tokens)>;

    NeuralSelfHealer(const HealingConfig& cfg = {});

    // Registra as funções que o healer vai usar para gerar e avaliar
    void set_token_decoder(TokenDecoder fn)    { decoder_   = std::move(fn); }
    void set_logits_extractor(LogitsExtractor fn) { extractor_ = std::move(fn); }
    void set_custom_validator(CustomValidator fn) {
        custom_validator_ = std::move(fn);
        validation_type_  = ValidationType::CUSTOM;
    }
    void set_validation_type(ValidationType vt) { validation_type_ = vt; }

    // Gera tokens com auto-correção ativa
    // Retorna os tokens gerados e preenche o report com diagnóstico
    std::vector<int> generate_healed(
        const std::vector<int>& prompt,
        int max_tokens,
        HealingReport& report,
        Context* ctx = nullptr
    );

    // Verifica se uma sequência já gerada passa nos critérios
    bool verify(const std::vector<int>& tokens, HealingReport& report);

private:
    HealingConfig       cfg_;
    ValidationType      validation_type_ = ValidationType::NONE;
    TokenDecoder        decoder_;
    LogitsExtractor     extractor_;
    CustomValidator     custom_validator_;

    // === Mecanismos internos ===

    // 1. ConfidenceGate: analisa a distribuição do último logit
    //    Retorna (max_prob, shanon_entropy)
    std::pair<float, float> measure_confidence(const Tensor& logits) const;

    // 2. ConsistencyCheck: gera N amostras e verifica concordância no primeiro token
    float measure_consistency(
        const std::vector<int>& prompt,
        float temperature,
        int samples
    ) const;

    // 3. SemanticValidator: verifica estrutura sintática
    bool validate_structure(const std::vector<int>& tokens) const;

    // Helper: valida balanceamento de brackets em sequência de token IDs
    // (Simplified: verifica contagem de tokens especiais via ID ranges)
    bool check_balanced_brackets(const std::vector<int>& tokens) const;
    bool check_json_structure(const std::vector<int>& tokens) const;
};

// Model-level entry point: wire the self-healer onto a JambaModel (logits
// extractor + D2FDecoder token decoder) and generate with active confidence/
// consistency/structure correction.  This makes the healer reachable from real
// inference, not just ad-hoc in tests.
std::vector<int> generate_with_self_healing(JambaModel& model,
                                            const std::vector<int>& prompt,
                                            int max_tokens,
                                            HealingReport& report,
                                            const HealingConfig& cfg = {});

} // namespace nsos
