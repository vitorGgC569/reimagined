#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nsos::optimizer_policy {

// One descriptor owns a contiguous interval from exactly one tensor. The
// bounds keep descriptor memory finite for larger models while allowing tests
// to exercise true multi-chunk tensors.
inline constexpr std::uint32_t kDefaultOptimizerChunkElements = 8192u;
inline constexpr std::uint32_t kMinimumOptimizerChunkElements = 32u;
inline constexpr std::uint32_t kMaximumOptimizerChunkElements =
    1u << 20;

#if defined(_WIN32) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

// std::getenv is safe here: names are compile-time constants, values are read
// only while the runtime policy lease prevents concurrent policy mutation, and
// no returned pointer survives the expression that consumes it. UCRT marks the
// standard function deprecated globally, so suppress that platform-only warning
// locally instead of allocating with _dupenv_s on every optimizer step.
inline bool parse_optimizer_boolean(const char* environment_name,
                                    bool default_value) {
    const char* value = std::getenv(environment_name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    if (value[0] == '0' && value[1] == '\0') return false;
    if (value[0] == '1' && value[1] == '\0') return true;
    throw std::invalid_argument(
        std::string(environment_name) + " must be exactly 0 or 1");
}

inline bool deterministic_adamw_chunked_enabled() {
    return parse_optimizer_boolean(
        "NSOS_DETERMINISTIC_ADAMW_CHUNKED", true);
}

inline bool device_sparse_adam_enabled() {
    return parse_optimizer_boolean("NSOS_MOE_DEVICE_ADAM", false);
}
inline constexpr const char* kDeviceSparseGradientIdentity =
    "explicit_group_contribution_device_v1";

inline bool device_gradient_clip_enabled() {
    return parse_optimizer_boolean("NSOS_DEVICE_GRAD_CLIP", false);
}

inline bool deterministic_finite_gate_deferred_enabled() noexcept {
    const char* value =
        std::getenv("NSOS_DETERMINISTIC_FINITE_GATE_DEFERRED");
    // Experimental after neutral A/B on the reference AMD workstation. Only
    // exact opt-in enables it; the separately synchronized gate is production.
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline bool optimizer_finite_chunked_enabled() {
    return parse_optimizer_boolean(
        "NSOS_OPTIMIZER_FINITE_CHUNKED", true);
}

inline std::uint32_t parse_optimizer_chunk_elements(
    const char* environment_name,
    std::uint32_t default_value = kDefaultOptimizerChunkElements) {
    const char* text = std::getenv(environment_name);
    if (text == nullptr || text[0] == '\0') {
        return default_value;
    }

    std::uint64_t parsed = 0;
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            throw std::invalid_argument(
                std::string(environment_name) +
                " must be a decimal power of two in [32, 1048576]");
        }
        const std::uint64_t digit =
            static_cast<std::uint64_t>(*cursor - '0');
        if (parsed >
            (static_cast<std::uint64_t>(
                 kMaximumOptimizerChunkElements) - digit) /
                10u) {
            throw std::invalid_argument(
                std::string(environment_name) + " exceeds 1048576");
        }
        parsed = parsed * 10u + digit;
    }

    if (parsed < kMinimumOptimizerChunkElements ||
        parsed > kMaximumOptimizerChunkElements ||
        (parsed & (parsed - 1u)) != 0u) {
        throw std::invalid_argument(
            std::string(environment_name) +
            " must be a power of two in [32, 1048576]");
    }
    return static_cast<std::uint32_t>(parsed);
}

inline std::uint32_t deterministic_adamw_chunk_elements() {
    return parse_optimizer_chunk_elements(
        "NSOS_DETERMINISTIC_ADAMW_CHUNK_ELEMENTS");
}

inline std::uint32_t optimizer_finite_chunk_elements() {
    return parse_optimizer_chunk_elements(
        "NSOS_OPTIMIZER_FINITE_CHUNK_ELEMENTS");
}

#if defined(_WIN32) && defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace nsos::optimizer_policy
