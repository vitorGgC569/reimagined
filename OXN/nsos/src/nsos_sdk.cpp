#include "../include/nsos_sdk.h"
#include "../include/nsos/sha256.h"
#include "../include/nsos/determinism.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <mutex>
#include <map>
#include <optional>
#include <random>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include "../include/gpu_backend.h"
#include "../include/cuda/device_buffer.h"
#include "../include/cuda/kernels.cuh"
#include "../include/cuda/pinned_buffer.h"
#endif

namespace nsos {

namespace {

std::string trim_copy(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::optional<long long> parse_integer(const std::string& text) {
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    try {
        size_t consumed = 0;
        const long long value = std::stoll(trimmed, &consumed);
        if (consumed != trimmed.size()) {
            return std::nullopt;
        }
        return value;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

bool environment_flag(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw) {
        return false;
    }
    std::string value = trim_copy(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (value == "1" || value == "true" || value == "yes" ||
        value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" ||
        value == "off" || value.empty()) {
        return false;
    }
    throw std::runtime_error(std::string("Invalid boolean value for ") + name);
}

constexpr uintmax_t kMaxConfigFileBytes = 1024 * 1024;
constexpr uintmax_t kMaxTokenizerPackBytes = 256ull * 1024ull * 1024ull;
constexpr uintmax_t kMaxEdgePackBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr uintmax_t kMaxWeightsPackBytes = 16ull * 1024ull * 1024ull * 1024ull;

std::filesystem::path atomic_temp_path(const std::filesystem::path& path) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t timestamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t ordinal =
        sequence.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    const uint64_t process_id = static_cast<uint64_t>(GetCurrentProcessId());
#else
    const uint64_t process_id = static_cast<uint64_t>(::getpid());
#endif
    return path.parent_path() /
           (path.filename().string() + ".tmp." +
            std::to_string(process_id) + "." +
            std::to_string(timestamp) + "." +
            std::to_string(ordinal));
}

void sync_file_to_storage(const std::filesystem::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Could not open file for durable flush: " + path.string());
    }
    const BOOL flushed = FlushFileBuffers(handle);
    const DWORD error = flushed ? ERROR_SUCCESS : GetLastError();
    CloseHandle(handle);
    if (!flushed) {
        throw std::runtime_error("Could not durably flush file: " + path.string() +
                                 " (Win32 error " + std::to_string(error) + ")");
    }
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error("Could not open file for durable flush: " + path.string());
    }
    const int result = ::fsync(descriptor);
    const int saved_errno = errno;
    ::close(descriptor);
    if (result != 0) {
        throw std::runtime_error("Could not durably flush file: " + path.string() +
                                 ": " + std::strerror(saved_errno));
    }
#endif
}

void sync_parent_directory(const std::filesystem::path& path) {
#ifndef _WIN32
    const std::filesystem::path parent = path.parent_path().empty()
                                             ? std::filesystem::path(".")
                                             : path.parent_path();
    const int descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        throw std::runtime_error("Could not open directory for durable flush: " +
                                 parent.string());
    }
    const int result = ::fsync(descriptor);
    const int saved_errno = errno;
    ::close(descriptor);
    if (result != 0) {
        throw std::runtime_error("Could not durably flush directory: " + parent.string() +
                                 ": " + std::strerror(saved_errno));
    }
#else
    (void)path;
    // MoveFileExW(..., MOVEFILE_WRITE_THROUGH) below supplies the corresponding
    // Windows durability guarantee for the rename and directory metadata.
#endif
}

void replace_file(const std::filesystem::path& temp_path, const std::filesystem::path& final_path) {
    try {
        sync_file_to_storage(temp_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_path, ignored);
        throw;
    }
#ifdef _WIN32
    if (!MoveFileExW(temp_path.c_str(), final_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::filesystem::remove(temp_path);
        throw std::runtime_error(
            "Could not atomically replace file: " + final_path.string() +
            " (Win32 error " + std::to_string(error) + ")");
    }
#else
    std::error_code ec;
    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path);
        throw std::runtime_error("Could not atomically replace file: " + final_path.string() +
                                 ": " + ec.message());
    }
#endif
    sync_parent_directory(final_path);
}

void ensure_regular_file_within_limit(const std::filesystem::path& path,
                                      uintmax_t max_bytes,
                                      const std::string& label) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
        throw std::runtime_error("Model pack " + label + " is missing or not a regular file");
    }
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("Could not read model pack " + label + " size");
    }
    if (size > max_bytes) {
        throw std::runtime_error("Model pack " + label + " exceeds configured size limit");
    }
}

bool path_has_parent_traversal(const std::filesystem::path& path) {
    for (const auto& part : path) {
        if (part == "..") {
            return true;
        }
    }
    return false;
}

std::filesystem::path pack_child_path(const std::filesystem::path& pack_root,
                                      const std::string& manifest_value,
                                      const std::string& label) {
    namespace fs = std::filesystem;
    fs::path child(manifest_value);
    if (child.empty() || child.is_absolute() || path_has_parent_traversal(child)) {
        throw std::runtime_error("Model pack manifest contains unsafe " + label + " path");
    }
    std::error_code ec;
    const fs::path canonical_root = fs::weakly_canonical(pack_root, ec);
    if (ec) {
        throw std::runtime_error("Could not canonicalize model pack root");
    }
    const fs::path candidate = fs::weakly_canonical(pack_root / child, ec);
    if (ec) {
        throw std::runtime_error("Could not canonicalize model pack " + label + " path");
    }
    const fs::path relative = candidate.lexically_relative(canonical_root);
    if (relative.empty() || relative.is_absolute() || path_has_parent_traversal(relative)) {
        throw std::runtime_error("Model pack " + label + " resolves outside the pack root");
    }
    return candidate;
}

std::unordered_map<std::string, std::string> read_key_value_file(const std::filesystem::path& path) {
    ensure_regular_file_within_limit(path, kMaxConfigFileBytes, path.filename().string());
    std::unordered_map<std::string, std::string> values;
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open config/manifest file: " + path.string());
    }

    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto sep = line.find('=');
        if (sep == std::string::npos) {
            throw std::runtime_error(
                "Malformed key/value line " + std::to_string(line_number) +
                " in " + path.string());
        }
        const std::string key = trim_copy(line.substr(0, sep));
        if (key.empty()) {
            throw std::runtime_error(
                "Empty key on line " + std::to_string(line_number) +
                " in " + path.string());
        }
        const auto [unused, inserted] = values.emplace(
            key, trim_copy(line.substr(sep + 1)));
        if (!inserted) {
            throw std::runtime_error(
                "Duplicate key '" + key + "' in " + path.string());
        }
    }
    return values;
}

void write_key_value_file(const std::filesystem::path& path,
                          const std::vector<std::pair<std::string, std::string>>& rows) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temp_path = atomic_temp_path(path);
    std::ofstream output(temp_path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Could not write file: " + temp_path.string());
    }
    for (const auto& [key, value] : rows) {
        output << key << "=" << value << "\n";
    }
    output.close();
    if (!output) {
        std::filesystem::remove(temp_path);
        throw std::runtime_error("Could not flush file: " + temp_path.string());
    }
    replace_file(temp_path, path);
}

std::string format_float_round_trip(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "Cannot persist a non-finite ModelConfig float");
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
    if (!output) {
        throw std::runtime_error(
            "Could not format ModelConfig float");
    }
    return output.str();
}

std::string sha256_checksum_file(const std::filesystem::path& path) {
    return integrity::sha256_file(path);
}

const std::string& require_pack_sha256(
    const std::unordered_map<std::string, std::string>& manifest,
    const std::string& label, const std::string& sha_key) {
    const auto sha_it = manifest.find(sha_key);
    if (sha_it == manifest.end() || sha_it->second.size() != 64 ||
        !std::all_of(
            sha_it->second.begin(), sha_it->second.end(),
            [](unsigned char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f');
            })) {
        throw std::runtime_error(
            "Model pack " + label +
            " is missing a canonical SHA-256 digest");
    }
    return sha_it->second;
}

void verify_pack_file_digest(
    const std::unordered_map<std::string, std::string>& manifest,
    const std::filesystem::path& path,
    const std::string& label,
    const std::string& sha_key) {
    const std::string& expected =
        require_pack_sha256(manifest, label, sha_key);
    if (sha256_checksum_file(path) != expected) {
        throw std::runtime_error(
            "Model pack " + label + " sha256 mismatch");
    }
}

void write_model_config(const std::filesystem::path& path, const ModelConfig& config) {
    std::vector<std::pair<std::string,std::string>> fields = {
            {"architecture_schema_version",
             std::to_string(config.architecture_schema_version)},
            {"num_layers", std::to_string(config.num_layers)},
            {"d_model", std::to_string(config.d_model)},
            {"vocab_size", std::to_string(config.vocab_size)},
            {"n_heads", std::to_string(config.n_heads)},
            {"n_kv_heads", std::to_string(config.n_kv_heads)},
            {"sliding_window", std::to_string(config.sliding_window)},
            {"attention_period", std::to_string(config.attention_period)},
            {"attention_slot", std::to_string(config.attention_slot)},
            {"hybrid_composition",
             std::to_string(
                 static_cast<int>(config.hybrid_composition))},
            {"force_mamba_last_layer",
             config.force_mamba_last_layer ? "true" : "false"},
            {"faithful_attention_linears",
             config.faithful_attention_linears ? "true" : "false"},
            {"hybrid_mamba_gate_init",
             format_float_round_trip(config.hybrid_mamba_gate_init)},
            {"hybrid_attention_gate_init",
             format_float_round_trip(config.hybrid_attention_gate_init)},
            {"hybrid_ffn_gate_init",
             format_float_round_trip(config.hybrid_ffn_gate_init)},
            {"num_experts", std::to_string(config.num_experts)},
            {"num_experts_per_token", std::to_string(config.num_experts_per_token)},
            {"use_moe", config.use_moe ? "true" : "false"},
            {"moe_period", std::to_string(config.moe_period)},
            {"moe_slot", std::to_string(config.moe_slot)},
            {"use_ttt", config.use_ttt ? "true" : "false"},
            {"ttt_period", std::to_string(config.ttt_period)},
            {"ttt_slot", std::to_string(config.ttt_slot)},
            {"use_gradient_checkpointing", config.use_gradient_checkpointing ? "true" : "false"},
            {"dropout", format_float_round_trip(config.dropout)},
            {"mcts_simulations", std::to_string(config.mcts_simulations)},
            {"mcts_depth", std::to_string(config.mcts_depth)},
            // A model pack is self-contained; persisting a machine-local
            // checkpoint path leaks host layout and can inject config lines.
            {"checkpoint_path", ""},
            {"max_context_tokens", std::to_string(config.max_context_tokens)},
            {"default_batch_size", std::to_string(config.default_batch_size)},
            {"use_cuda", config.use_cuda ? "true" : "false"},
            {"use_exact_attention_training", config.use_exact_attention_training ? "true" : "false"},
            {"use_flash_attn", config.use_flash_attn ? "true" : "false"},
            // Architecture fields previously MISSING from the round-trip: a pack
            // saved with any of these non-default silently flipped back to the
            // default on load (the checkpoint-v2 fingerprint catches proper/tie
            // loudly, but rope_theta/KAN/CHRASS/slender/expert-hidden would flip
            // silently).  Full set now persisted.
            {"moe_expert_hidden_dim", std::to_string(config.moe_expert_hidden_dim)},
            {"use_kan", config.use_kan ? "true" : "false"},
            {"use_chrass", config.use_chrass ? "true" : "false"},
            {"chrass_density",
             format_float_round_trip(config.chrass_density)},
            {"chrass_seed", std::to_string(config.chrass_seed)},
            {"logit_l2_beta",
             format_float_round_trip(config.logit_l2_beta)},
            {"pantheon_vib_beta",
             format_float_round_trip(config.pantheon_vib_beta)},
            {"use_slender_embedding", config.use_slender_embedding ? "true" : "false"},
            {"mamba_proper_ssm", config.mamba_proper_ssm ? "true" : "false"},
            {"mamba_state_expansion", config.mamba_state_expansion ? "true" : "false"},
            {"mamba_d_state", std::to_string(config.mamba_d_state)},
            {"mamba_conv_kernel", std::to_string(config.mamba_conv_kernel)},
            {"mamba2_faithful", config.mamba2_faithful ? "true" : "false"},
            {"mamba_expand", std::to_string(config.mamba_expand)},
            {"mamba_head_dim", std::to_string(config.mamba_head_dim)},
            {"mamba_n_groups", std::to_string(config.mamba_n_groups)},
            {"tie_word_embeddings", config.tie_word_embeddings ? "true" : "false"},
            {"rope_theta", format_float_round_trip(config.rope_theta)},
        };
    if(config.architecture_schema_version>=3) {
        fields.emplace_back("mamba3_enabled", config.mamba3_enabled ? "true" : "false");
        fields.emplace_back("mamba3_schema_version", std::to_string(config.mamba3_schema_version));
        fields.emplace_back("mamba3_state_dim", std::to_string(config.mamba3_state_dim));
        fields.emplace_back("mamba3_mimo", config.mamba3_mimo ? "true" : "false");
        fields.emplace_back("mamba3_mimo_rank", std::to_string(config.mamba3_mimo_rank));
        fields.emplace_back("mamba3_outproj_norm", config.mamba3_outproj_norm ? "true" : "false");
        fields.emplace_back("mamba3_rope_fraction", format_float_round_trip(config.mamba3_rope_fraction));
        fields.emplace_back("mamba3_norm_eps", format_float_round_trip(config.mamba3_norm_eps));
        fields.emplace_back("mamba3_a_floor", format_float_round_trip(config.mamba3_a_floor));
    }
    write_key_value_file(path, fields);
}

ModelConfig read_model_config(const std::filesystem::path& path) {
    const auto values = read_key_value_file(path);
    ModelConfig config;
    auto get_int = [&](const std::string& key, int fallback) {
        const auto it = values.find(key);
        if (it == values.end()) return fallback;
        const auto parsed = parse_integer(it->second);
        if (!parsed || *parsed < std::numeric_limits<int>::min() ||
            *parsed > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Invalid integer ModelConfig." + key);
        }
        return static_cast<int>(*parsed);
    };
    auto get_float = [&](const std::string& key, float fallback) {
        const auto it = values.find(key);
        if (it == values.end()) return fallback;
        try {
            const std::string value = trim_copy(it->second);
            size_t consumed = 0;
            const float parsed = std::stof(value, &consumed);
            if (consumed != value.size()) {
                throw std::runtime_error("trailing characters");
            }
            return parsed;
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid floating-point ModelConfig." + key);
        }
    };
    auto get_bool = [&](const std::string& key, bool fallback) {
        const auto it = values.find(key);
        if (it == values.end()) return fallback;
        std::string value = trim_copy(it->second);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (value == "1" || value == "true") return true;
        if (value == "0" || value == "false") return false;
        throw std::runtime_error("Invalid boolean ModelConfig." + key);
    };
    auto get_u32 = [&](const std::string& key, uint32_t fallback) {
        const auto it = values.find(key);
        if (it == values.end()) return fallback;
        try {
            const std::string value = trim_copy(it->second);
            size_t consumed = 0;
            const unsigned long long parsed = std::stoull(value, &consumed);
            if (consumed != value.size() ||
                parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("out of range");
            }
            return static_cast<uint32_t>(parsed);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid uint32 ModelConfig." + key);
        }
    };
    auto get_string = [&](const std::string& key, const std::string& fallback) {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    };

    const bool has_architecture_schema =
        values.count("architecture_schema_version") != 0;
    config.architecture_schema_version =
        get_int("architecture_schema_version",
                has_architecture_schema
                    ? config.architecture_schema_version
                    : 1);
    // Schema v2 is a complete reproducibility contract, not a bag of optional
    // defaults. A producer that omits any persisted field must fail closed;
    // otherwise a newer reader can silently construct a different model or
    // training policy while the file still claims the current schema.
    const std::initializer_list<const char*> schema_v2_fields = {
        "architecture_schema_version",
        "num_layers",
        "d_model",
        "vocab_size",
        "n_heads",
        "n_kv_heads",
        "sliding_window",
        "attention_period",
        "attention_slot",
        "hybrid_composition",
        "force_mamba_last_layer",
        "faithful_attention_linears",
        "hybrid_mamba_gate_init",
        "hybrid_attention_gate_init",
        "hybrid_ffn_gate_init",
        "num_experts",
        "num_experts_per_token",
        "use_moe",
        "moe_period",
        "moe_slot",
        "use_ttt",
        "ttt_period",
        "ttt_slot",
        "use_gradient_checkpointing",
        "dropout",
        "mcts_simulations",
        "mcts_depth",
        "checkpoint_path",
        "max_context_tokens",
        "default_batch_size",
        "use_cuda",
        "use_exact_attention_training",
        "use_flash_attn",
        "moe_expert_hidden_dim",
        "use_kan",
        "use_chrass",
        "chrass_density",
        "chrass_seed",
        "logit_l2_beta",
        "pantheon_vib_beta",
        "use_slender_embedding",
        "mamba_proper_ssm",
        "mamba_state_expansion",
        "mamba_d_state",
        "mamba_conv_kernel",
        "mamba2_faithful",
        "mamba_expand",
        "mamba_head_dim",
        "mamba_n_groups",
        "tie_word_embeddings",
        "rope_theta",
    };
    if (config.architecture_schema_version >= 2) {
        std::unordered_set<std::string> allowed_fields;
        allowed_fields.reserve(schema_v2_fields.size() * 2);
        for (const char* field : schema_v2_fields) {
            allowed_fields.emplace(field);
            if (values.count(field) == 0) {
                throw std::runtime_error(
                    std::string("ModelConfig schema v2 is missing required field: ") +
                    field);
            }
        }
        if(config.architecture_schema_version>=3) {
            for(const char* field: {"mamba3_enabled","mamba3_schema_version","mamba3_state_dim","mamba3_mimo","mamba3_mimo_rank","mamba3_outproj_norm","mamba3_rope_fraction","mamba3_norm_eps","mamba3_a_floor"}) {
                allowed_fields.emplace(field);
                if(!values.count(field)) throw std::runtime_error(std::string("ModelConfig schema v3 missing required field: ")+field);
            }
        }
        for (const auto& [field, _] : values) {
            if (allowed_fields.count(field) == 0) {
                throw std::runtime_error(
                    "ModelConfig schema v2 contains unknown field: " +
                    field);
            }
        }
    }
    config.num_layers = get_int("num_layers", config.num_layers);
    config.d_model = get_int("d_model", config.d_model);
    config.vocab_size = get_int("vocab_size", config.vocab_size);
    config.n_heads = get_int("n_heads", config.n_heads);
    config.n_kv_heads = get_int("n_kv_heads", config.n_kv_heads);
    config.sliding_window = get_int("sliding_window", config.sliding_window);
    config.attention_period = get_int("attention_period", config.attention_period);
    config.attention_slot = get_int("attention_slot", config.attention_slot);
    if (config.architecture_schema_version == 1) {
        config.hybrid_composition =
            HybridComposition::LegacyReplacement;
        config.force_mamba_last_layer = true;
        config.faithful_attention_linears = false;
    } else {
        config.hybrid_composition = static_cast<HybridComposition>(
            get_int("hybrid_composition",
                    static_cast<int>(config.hybrid_composition)));
        config.force_mamba_last_layer = get_bool(
            "force_mamba_last_layer",
            config.force_mamba_last_layer);
        config.faithful_attention_linears = get_bool(
            "faithful_attention_linears",
            config.faithful_attention_linears);
    }
    config.hybrid_mamba_gate_init =
        get_float("hybrid_mamba_gate_init",
                  config.hybrid_mamba_gate_init);
    config.hybrid_attention_gate_init =
        get_float("hybrid_attention_gate_init",
                  config.hybrid_attention_gate_init);
    config.hybrid_ffn_gate_init =
        get_float("hybrid_ffn_gate_init",
                  config.hybrid_ffn_gate_init);
    config.num_experts = get_int("num_experts", config.num_experts);
    config.num_experts_per_token = get_int("num_experts_per_token", config.num_experts_per_token);
    config.use_moe = get_bool("use_moe", config.use_moe);
    config.moe_period = get_int("moe_period", config.moe_period);
    config.moe_slot = get_int("moe_slot", config.moe_slot);
    config.use_ttt = get_bool("use_ttt", config.use_ttt);
    config.ttt_period = get_int("ttt_period", config.ttt_period);
    config.ttt_slot = get_int("ttt_slot", config.ttt_slot);
    config.use_gradient_checkpointing =
        get_bool("use_gradient_checkpointing", config.use_gradient_checkpointing);
    config.dropout = get_float("dropout", config.dropout);
    config.mcts_simulations = get_int("mcts_simulations", config.mcts_simulations);
    config.mcts_depth = get_int("mcts_depth", config.mcts_depth);
    config.checkpoint_path = get_string("checkpoint_path", config.checkpoint_path);
    config.max_context_tokens = get_int("max_context_tokens", config.max_context_tokens);
    config.default_batch_size = get_int("default_batch_size", config.default_batch_size);
    config.use_cuda = get_bool("use_cuda", config.use_cuda);
    config.use_exact_attention_training =
        get_bool("use_exact_attention_training", config.use_exact_attention_training);
    config.use_flash_attn = get_bool("use_flash_attn", config.use_flash_attn);
    config.moe_expert_hidden_dim =
        get_int("moe_expert_hidden_dim", config.moe_expert_hidden_dim);
    config.use_kan = get_bool("use_kan", config.use_kan);
    config.use_chrass = get_bool("use_chrass", config.use_chrass);
    config.chrass_density = get_float("chrass_density", config.chrass_density);
    config.chrass_seed = get_u32("chrass_seed", config.chrass_seed);
    config.logit_l2_beta = get_float("logit_l2_beta", config.logit_l2_beta);
    config.pantheon_vib_beta = get_float("pantheon_vib_beta", config.pantheon_vib_beta);
    config.use_slender_embedding =
        get_bool("use_slender_embedding", config.use_slender_embedding);
    config.mamba_proper_ssm = get_bool("mamba_proper_ssm", config.mamba_proper_ssm);
    config.mamba_state_expansion =
        get_bool("mamba_state_expansion", config.mamba_state_expansion);
    config.mamba_d_state = get_int("mamba_d_state", config.mamba_d_state);
    config.mamba_conv_kernel = get_int("mamba_conv_kernel", config.mamba_conv_kernel);
    // Packs written before the faithful block existed must keep the old
    // parameter layout even though new ModelConfig instances default to it.
    config.mamba2_faithful =
        get_bool("mamba2_faithful",
                 values.count("mamba2_faithful") != 0
                     ? config.mamba2_faithful
                     : false);
    config.mamba_expand = get_int("mamba_expand", config.mamba_expand);
    config.mamba_head_dim = get_int("mamba_head_dim", config.mamba_head_dim);
    config.mamba_n_groups = get_int("mamba_n_groups", config.mamba_n_groups);
    config.mamba3_enabled = get_bool("mamba3_enabled", config.mamba3_enabled);
    config.mamba3_schema_version = get_int("mamba3_schema_version", config.mamba3_schema_version);
    config.mamba3_state_dim = get_int("mamba3_state_dim", config.mamba3_state_dim);
    config.mamba3_mimo = get_bool("mamba3_mimo", config.mamba3_mimo);
    config.mamba3_mimo_rank = get_int("mamba3_mimo_rank", config.mamba3_mimo_rank);
    config.mamba3_outproj_norm = get_bool("mamba3_outproj_norm", config.mamba3_outproj_norm);
    config.mamba3_rope_fraction = get_float("mamba3_rope_fraction", config.mamba3_rope_fraction);
    config.mamba3_norm_eps = get_float("mamba3_norm_eps", config.mamba3_norm_eps);
    config.mamba3_a_floor = get_float("mamba3_a_floor", config.mamba3_a_floor);
    config.tie_word_embeddings =
        get_bool("tie_word_embeddings", config.tie_word_embeddings);
    config.rope_theta = get_float("rope_theta", config.rope_theta);
    return config;
}

void validate_model_config_for_pack(const ModelConfig& config) {
    validate_model_config(config);
}

void apply_model_load_options(ModelConfig& base,
                              const ModelLoadOptions& options) {
    if (options.use_cuda) base.use_cuda = *options.use_cuda;
    if (options.default_batch_size) {
        base.default_batch_size = *options.default_batch_size;
    }
    if (options.mcts_simulations) {
        base.mcts_simulations = *options.mcts_simulations;
    }
    if (options.mcts_depth) base.mcts_depth = *options.mcts_depth;
    if (options.checkpoint_path) {
        base.checkpoint_path = *options.checkpoint_path;
    }
}

std::string decode_token_piece(const Tokenizer& tokenizer, int token) {
    const std::string piece = tokenizer.decode({token});
    if (!piece.empty()) {
        return piece;
    }
    const int clamped = std::clamp(token, 0, 255);
    if (clamped < 0x80) {
        return std::string(1, static_cast<char>(clamped));
    }
    return "\xEF\xBF\xBD";
}

uint64_t fnv1a_hash_text(const std::string& text) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (unsigned char byte : text) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= kPrime;
    }
    return hash;
}

std::mt19937 make_sampler_rng(const std::string& lane, uint64_t sequence_id) {
    auto seeded =
        determinism::DeterminismManager::instance().get_rng_for_operation("nsos_sdk", lane,
                                                                          sequence_id);
    return std::mt19937(static_cast<uint32_t>(seeded()));
}

struct SamplerWorkspace {
    std::vector<float> scaled;
    std::vector<int> candidate_indices;
    std::vector<float> candidate_probs;
    std::vector<char> seen;
    std::vector<int> banned;
};

float bounded_logit(float value) {
    if (std::isnan(value)) {
        return -1.0e30f;
    }
    if (value == std::numeric_limits<float>::infinity()) {
        return 1.0e30f;
    }
    if (value == -std::numeric_limits<float>::infinity()) {
        return -1.0e30f;
    }
    return std::clamp(value, -1.0e30f, 1.0e30f);
}

void validate_generation_options(const GenerationOptions& options,
                                 int vocab_size) {
    constexpr int kAbsoluteMaxGeneratedTokens = 1 << 20;
    constexpr int kAbsoluteMaxContextTokens = 1 << 24;
    if (options.max_tokens < 0 || options.max_tokens > kAbsoluteMaxGeneratedTokens) {
        throw std::invalid_argument("max_tokens is outside the supported range");
    }
    if (options.min_new_tokens < 0 ||
        options.min_new_tokens > kAbsoluteMaxGeneratedTokens) {
        throw std::invalid_argument("min_new_tokens is outside the supported range");
    }
    if (!std::isfinite(options.temperature) || options.temperature < 0.0f ||
        options.temperature > 100.0f) {
        throw std::invalid_argument("temperature must be finite and between 0 and 100");
    }
    if (!std::isfinite(options.top_p) || options.top_p <= 0.0f ||
        options.top_p > 1.0f) {
        throw std::invalid_argument("top_p must be finite and in (0, 1]");
    }
    if (options.top_k < 0 || (vocab_size > 0 && options.top_k > vocab_size)) {
        throw std::invalid_argument("top_k must be between 0 and the vocabulary size");
    }
    if (options.eos_token_id < 0 || options.eos_token_id >= vocab_size) {
        throw std::invalid_argument("eos_token_id is outside the model vocabulary");
    }
    if (options.max_context_tokens > kAbsoluteMaxContextTokens) {
        throw std::invalid_argument("max_context_tokens exceeds the supported range");
    }
    if (!std::isfinite(options.repetition_penalty) ||
        options.repetition_penalty < 1.0f || options.repetition_penalty > 100.0f) {
        throw std::invalid_argument(
            "repetition_penalty must be finite and between 1 and 100");
    }
    if (options.no_repeat_ngram_size < 0 || options.no_repeat_ngram_size > 128) {
        throw std::invalid_argument("no_repeat_ngram_size must be between 0 and 128");
    }
}

struct StreamingInferenceGuard {
    JambaModel* model = nullptr;
    std::atomic<bool>* poisoned = nullptr;
    void restore() {
        if (model != nullptr) {
            model->set_streaming_inference(false);
            model = nullptr;
        }
    }
    ~StreamingInferenceGuard() {
        if (model != nullptr) {
            try {
                model->set_streaming_inference(false);
            } catch (...) {
                // Preserve the in-flight exception, but make every subsequent
                // engine operation fail-stop until a fresh model is loaded.
                if (poisoned) {
                    poisoned->store(true, std::memory_order_release);
                }
            }
        }
    }
};

int select_best_token_fallback(const std::vector<float>& scaled,
                               int vocab_size,
                               int eos_token_id,
                               bool suppress_control_tokens,
                               const std::function<bool(int, int)>& is_control_token) {
    int best = -1;
    float best_value = -1e30f;
    for (int token = 0; token < vocab_size; ++token) {
        if (token == eos_token_id) {
            continue;
        }
        if (suppress_control_tokens && is_control_token(token, vocab_size)) {
            continue;
        }
        const float value = bounded_logit(scaled[static_cast<size_t>(token)]);
        if (value > best_value) {
            best_value = value;
            best = token;
        }
    }
    return best >= 0 ? best : (eos_token_id >= 0 ? eos_token_id : 0);
}

// Single source of truth for the no-repeat-ngram banned set, used by both the
// host sampler (greedy + stochastic branches) and the GPU greedy sampler.
// Fills `banned` with the tokens that would complete a repeated n-gram given the
// current generated suffix; clears it when there is not enough history.  Extracted
// verbatim from the previously-inlined logic so host and GPU never drift.
void compute_no_repeat_ngram_banned(const std::vector<int>& output_tokens,
                                    size_t prompt_tokens_used,
                                    int no_repeat_ngram_size,
                                    size_t generated_so_far,
                                    std::vector<int>& banned) {
    banned.clear();
    if (generated_so_far == 0 || no_repeat_ngram_size <= 1) {
        return;
    }
    const int ngram = no_repeat_ngram_size;
    if (generated_so_far + 1 < static_cast<size_t>(ngram)) {
        return;
    }
    if (ngram == 2) {
        const int last_token = output_tokens.back();
        for (size_t i = prompt_tokens_used; i + 1 < output_tokens.size(); ++i) {
            if (output_tokens[i] == last_token) {
                banned.push_back(output_tokens[i + 1]);
            }
        }
    } else {
        const size_t prefix_len = static_cast<size_t>(ngram - 1);
        const size_t prefix_start = output_tokens.size() - prefix_len;
        for (size_t i = prompt_tokens_used;
             i + static_cast<size_t>(ngram) <= output_tokens.size(); ++i) {
            bool prefix_match = true;
            for (size_t j = 0; j < prefix_len; ++j) {
                if (output_tokens[i + j] != output_tokens[prefix_start + j]) {
                    prefix_match = false;
                    break;
                }
            }
            if (prefix_match) {
                banned.push_back(output_tokens[i + prefix_len]);
            }
        }
    }
}

int sample_from_host_logits_row(const float* raw,
                                int vocab_size,
                                int top_k,
                                const GenerationOptions& options,
                                const std::vector<int>& output_tokens,
                                size_t prompt_tokens_used,
                                std::mt19937& rng,
                                SamplerWorkspace& workspace,
                                const std::function<bool(int, int)>& is_control_token,
                                double* sampler_ms_accum) {
    const auto sampler_started = std::chrono::steady_clock::now();
    if (vocab_size <= 0 || raw == nullptr) {
        return options.eos_token_id;
    }

    const size_t generated_so_far =
        output_tokens.size() > prompt_tokens_used ? (output_tokens.size() - prompt_tokens_used) : 0;

    const bool block_eos =
        generated_so_far < static_cast<size_t>(std::max(options.min_new_tokens, 0)) &&
        options.eos_token_id >= 0 && options.eos_token_id < vocab_size;
    const bool suppress_control =
        generated_so_far < static_cast<size_t>(std::max(options.min_new_tokens, 0)) &&
        options.suppress_control_tokens_at_start;
    const bool greedy_selection = options.temperature <= 1e-5f || top_k == 1;
    const bool top_p_can_change_selection =
        top_k != 1 && options.top_p < 1.0f && options.top_p > 0.0f;
    if (greedy_selection && !top_p_can_change_selection) {
        const auto sampler_started_fast = sampler_started;
        int best_token = -1;
        float best_value = -1e30f;
        workspace.seen.assign(static_cast<size_t>(vocab_size), 0);
        if (generated_so_far > 0 && options.no_repeat_ngram_size > 1) {
            compute_no_repeat_ngram_banned(output_tokens, prompt_tokens_used,
                                           options.no_repeat_ngram_size,
                                           generated_so_far, workspace.banned);
            for (int token : workspace.banned) {
                if (token >= 0 && token < vocab_size) {
                    workspace.seen[static_cast<size_t>(token)] = 1;
                }
            }
        }
        if (block_eos) {
            workspace.seen[static_cast<size_t>(options.eos_token_id)] = 1;
        }
        if (generated_so_far > 0 && options.repetition_penalty > 1.0f) {
            const float penalty = std::max(options.repetition_penalty, 1.0f);
            std::vector<char> repeated(static_cast<size_t>(vocab_size), 0);
            for (size_t index = prompt_tokens_used; index < output_tokens.size(); ++index) {
                const int token = output_tokens[index];
                if (token < 0 || token >= vocab_size || repeated[static_cast<size_t>(token)]) {
                    continue;
                }
                repeated[static_cast<size_t>(token)] = 1;
            }
            for (int token = 0; token < vocab_size; ++token) {
                if (workspace.seen[static_cast<size_t>(token)]) {
                    continue;
                }
                if (suppress_control && is_control_token(token, vocab_size)) {
                    continue;
                }
                float value = bounded_logit(raw[token]);
                if (repeated[static_cast<size_t>(token)]) {
                    if (value >= 0.0f) {
                        value /= penalty;
                    } else {
                        value *= penalty;
                    }
                }
                if (value > best_value) {
                    best_value = value;
                    best_token = token;
                }
            }
        } else {
            for (int token = 0; token < vocab_size; ++token) {
                if (workspace.seen[static_cast<size_t>(token)]) {
                    continue;
                }
                if (suppress_control && is_control_token(token, vocab_size)) {
                    continue;
                }
                const float value = bounded_logit(raw[token]);
                if (value > best_value) {
                    best_value = value;
                    best_token = token;
                }
            }
        }
        if (best_token < 0) {
            for (int token = 0; token < vocab_size; ++token) {
                if (token == options.eos_token_id && block_eos) {
                    continue;
                }
                if (suppress_control && is_control_token(token, vocab_size)) {
                    continue;
                }
                best_token = token;
                break;
            }
            if (best_token < 0) {
                best_token = options.eos_token_id >= 0 ? options.eos_token_id : 0;
            }
        }
        if (sampler_ms_accum) {
            *sampler_ms_accum +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - sampler_started_fast)
                                        .count()) /
                1000.0;
        }
        return best_token;
    }

    const float temp = std::max(options.temperature, 1e-6f);
    workspace.scaled.resize(static_cast<size_t>(vocab_size));
    for (int token = 0; token < vocab_size; ++token) {
        workspace.scaled[static_cast<size_t>(token)] = bounded_logit(raw[token]) / temp;
    }

    if (generated_so_far > 0 && options.repetition_penalty > 1.0f) {
        const float penalty = std::max(options.repetition_penalty, 1.0f);
        workspace.seen.assign(static_cast<size_t>(vocab_size), 0);
        for (size_t index = prompt_tokens_used; index < output_tokens.size(); ++index) {
            const int token = output_tokens[index];
            if (token < 0 || token >= vocab_size || workspace.seen[static_cast<size_t>(token)]) {
                continue;
            }
            workspace.seen[static_cast<size_t>(token)] = 1;
            if (workspace.scaled[static_cast<size_t>(token)] >= 0.0f) {
                workspace.scaled[static_cast<size_t>(token)] /= penalty;
            } else {
                workspace.scaled[static_cast<size_t>(token)] *= penalty;
            }
        }
    }

    workspace.candidate_indices.resize(static_cast<size_t>(vocab_size));
    std::iota(workspace.candidate_indices.begin(), workspace.candidate_indices.end(), 0);
    if (top_k > 0 && top_k < vocab_size) {
        std::nth_element(workspace.candidate_indices.begin(),
                         workspace.candidate_indices.begin() + top_k,
                         workspace.candidate_indices.end(),
                         [&](int lhs, int rhs) {
                             const float left = workspace.scaled[static_cast<size_t>(lhs)];
                             const float right = workspace.scaled[static_cast<size_t>(rhs)];
                             return left != right ? left > right : lhs < rhs;
                         });
        workspace.candidate_indices.resize(static_cast<size_t>(top_k));
    }

    workspace.seen.assign(static_cast<size_t>(vocab_size), 0);
    if (generated_so_far > 0 && options.no_repeat_ngram_size > 1) {
        compute_no_repeat_ngram_banned(output_tokens, prompt_tokens_used,
                                       options.no_repeat_ngram_size,
                                       generated_so_far, workspace.banned);
        for (int token : workspace.banned) {
            if (token >= 0 && token < vocab_size) {
                workspace.seen[static_cast<size_t>(token)] = 1;
            }
        }
    }

    if (block_eos) {
        workspace.seen[static_cast<size_t>(options.eos_token_id)] = 1;
    }

    auto candidate_end =
        std::remove_if(workspace.candidate_indices.begin(),
                       workspace.candidate_indices.end(),
                       [&](int token) {
                           if (token < 0 || token >= vocab_size) {
                               return true;
                           }
                           if (workspace.seen[static_cast<size_t>(token)]) {
                               return true;
                           }
                           if (suppress_control && is_control_token(token, vocab_size)) {
                               return true;
                           }
                           return false;
                       });
    workspace.candidate_indices.erase(candidate_end, workspace.candidate_indices.end());

    if (workspace.candidate_indices.empty()) {
        const int fallback = select_best_token_fallback(workspace.scaled,
                                                        vocab_size,
                                                        block_eos ? options.eos_token_id : -1,
                                                        suppress_control,
                                                        is_control_token);
        if (sampler_ms_accum) {
            *sampler_ms_accum +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - sampler_started)
                                        .count()) /
                1000.0;
        }
        return fallback;
    }

    if (options.top_p < 1.0f && options.top_p > 0.0f &&
        workspace.candidate_indices.size() > 1) {
        std::sort(workspace.candidate_indices.begin(),
                  workspace.candidate_indices.end(),
                  [&](int lhs, int rhs) {
                      const float left = workspace.scaled[static_cast<size_t>(lhs)];
                      const float right = workspace.scaled[static_cast<size_t>(rhs)];
                      return left != right ? left > right : lhs < rhs;
                  });
        const float max_val = workspace.scaled[static_cast<size_t>(workspace.candidate_indices.front())];
        float sum_exp = 0.0f;
        workspace.candidate_probs.resize(workspace.candidate_indices.size());
        for (size_t i = 0; i < workspace.candidate_indices.size(); ++i) {
            const int token = workspace.candidate_indices[i];
            const float value = std::exp(workspace.scaled[static_cast<size_t>(token)] - max_val);
            workspace.candidate_probs[i] = value;
            sum_exp += value;
        }
        const float inv_sum = 1.0f / std::max(sum_exp, 1e-9f);
        float cumulative = 0.0f;
        size_t keep_count = 0;
        for (size_t i = 0; i < workspace.candidate_indices.size(); ++i) {
            cumulative += workspace.candidate_probs[i] * inv_sum;
            ++keep_count;
            if (cumulative >= options.top_p) {
                break;
            }
        }
        keep_count = std::max<size_t>(keep_count, 1);
        workspace.candidate_indices.resize(keep_count);
    }

    float max_val = -1e30f;
    for (int token : workspace.candidate_indices) {
        max_val = std::max(max_val, workspace.scaled[static_cast<size_t>(token)]);
    }

    workspace.candidate_probs.resize(workspace.candidate_indices.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < workspace.candidate_indices.size(); ++i) {
        const int token = workspace.candidate_indices[i];
        const float value = std::exp(workspace.scaled[static_cast<size_t>(token)] - max_val);
        workspace.candidate_probs[i] = value;
        sum_exp += value;
    }

    if (sum_exp <= 1e-9f) {
        const int fallback = workspace.candidate_indices.front();
        if (sampler_ms_accum) {
            *sampler_ms_accum +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - sampler_started)
                                        .count()) /
                1000.0;
        }
        return fallback;
    }

    const float inv_sum = 1.0f / sum_exp;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float target = dist(rng);
    float cumulative = 0.0f;
    int selected = workspace.candidate_indices.back();
    for (size_t i = 0; i < workspace.candidate_indices.size(); ++i) {
        cumulative += workspace.candidate_probs[i] * inv_sum;
        if (target <= cumulative) {
            selected = workspace.candidate_indices[i];
            break;
        }
    }

    if (sampler_ms_accum) {
        *sampler_ms_accum +=
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - sampler_started)
                                    .count()) /
            1000.0;
    }
    return selected;
}

#ifdef USE_CUDA
bool gpu_greedy_sampler_enabled() {
    // GPU-first DEFAULT ON (NSOS_GPU_SAMPLER=0 opts out).  Greedy-only path
    // that mirrors the host branch exactly and falls back to it on ANY CUDA
    // error or fully-masked row (-1 sentinel), so the default is safe: worst
    // case is the historical host behaviour.
    static const bool enabled = [] {
        const char* v = std::getenv("NSOS_GPU_SAMPLER");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

// On-device greedy decode sampler (opt-in NSOS_GPU_SAMPLER).  Keeps the per-token
// selection on the GPU: a device "repeated" mask maintained incrementally, the
// per-token "seen" (no-repeat-ngram bans + blocked-EOS) mask, and a one-time
// "control" mask, fed to launch_decode_greedy_argmax.  Removes the per-token
// [vocab] D2H + host vocab scan and mirrors sample_from_host_logits_row's greedy
// branch exactly.  Returns false on any CUDA error so the caller falls back to
// the host path.  Single-threaded decode use (one instance per generation).
class GpuGreedySampler {
public:
    ~GpuGreedySampler() {
        // NSOS_D2H_TIMING=1: report the measured per-token D2H latency of
        // the pinned staging path used by this generation — the empirical
        // counterpart of nsos_bench_d2h_copy().
        if (d2h_copies_ > 0) {
            std::fprintf(stderr,
                         "[nsos] decode token D2H (%s staging): n=%zu "
                         "avg=%.2f us total=%.2f ms\n",
                         "pinned", d2h_copies_,
                         d2h_us_total_ / static_cast<double>(d2h_copies_),
                         d2h_us_total_ / 1000.0);
        }
    }

    bool select(const float* raw_row_device, int vocab,
                const GenerationOptions& options,
                const std::vector<int>& output_tokens, size_t prompt_tokens_used,
                const std::function<bool(int, int)>& is_control_token,
                int& out_token) {
        if (poisoned_ || vocab <= 0 || raw_row_device == nullptr) return false;
        if (!ensure(vocab)) return false;
        if (!control_built_ && !build_control(vocab, is_control_token)) return false;

        const size_t generated_so_far =
            output_tokens.size() > prompt_tokens_used
                ? (output_tokens.size() - prompt_tokens_used)
                : 0;
        if (marked_count_ > generated_so_far) {
            return poison_without_runtime_error(
                "generated-token history moved backwards");
        }

        // Finish host work that may allocate before the first asynchronous
        // write, so an allocation exception cannot strand an uncommitted
        // device mutation.
        banned_.clear();
        if (generated_so_far > 0 && options.no_repeat_ngram_size > 1) {
            compute_no_repeat_ngram_banned(output_tokens, prompt_tokens_used,
                                           options.no_repeat_ngram_size,
                                           generated_so_far, banned_);
        }

        // repeated mask: incrementally mark the generated tokens (output[prompt:])
        // so the penalty matches the host's per-token "repeated" set.
        for (size_t i = prompt_tokens_used + marked_count_;
             i < output_tokens.size(); ++i) {
            const int t = output_tokens[i];
            if (t >= 0 && t < vocab) {
                const cudaError_t status =
                    cudaMemsetAsync(d_repeated_.get() + t, 1, 1, nsos::gpu::current_stream());
                if (status != cudaSuccess) {
                    return poison_runtime(status, "repetition-mask update");
                }
            }
        }

        // seen mask: zero, then scatter no-repeat-ngram bans + blocked-EOS.
        cudaError_t status = cudaMemsetAsync(
            d_seen_.get(), 0, static_cast<size_t>(vocab), nsos::gpu::current_stream());
        if (status != cudaSuccess) {
            return poison_runtime(status, "seen-mask reset");
        }
        for (int t : banned_) {
            if (t >= 0 && t < vocab) {
                status = cudaMemsetAsync(d_seen_.get() + t, 1, 1, nsos::gpu::current_stream());
                if (status != cudaSuccess) {
                    return poison_runtime(status, "seen-mask update");
                }
            }
        }
        const bool block_eos =
            generated_so_far <
                static_cast<size_t>(std::max(options.min_new_tokens, 0)) &&
            options.eos_token_id >= 0 && options.eos_token_id < vocab;
        if (block_eos) {
            status = cudaMemsetAsync(
                d_seen_.get() + options.eos_token_id, 1, 1, nsos::gpu::current_stream());
            if (status != cudaSuccess) {
                return poison_runtime(status, "EOS-mask update");
            }
        }
        const bool suppress_control =
            generated_so_far <
                static_cast<size_t>(std::max(options.min_new_tokens, 0)) &&
            options.suppress_control_tokens_at_start;
        const bool penalty_active =
            generated_so_far > 0 && options.repetition_penalty > 1.0f;
        const float penalty =
            penalty_active ? std::max(options.repetition_penalty, 1.0f) : 1.0f;

        launch_decode_greedy_argmax(raw_row_device, vocab,
                                    penalty_active
                                        ? d_repeated_.get()
                                        : nullptr,
                                    d_seen_.get(), d_control_.get(),
                                    suppress_control ? 1 : 0, penalty,
                                    d_result_.get());
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return poison_runtime(status, "greedy-argmax launch");
        }
        // The four-byte D2H uses owned pinned staging. A pageable stack
        // fallback could not be retained safely if completion became
        // ambiguous.
        static const bool d2h_timing = [] {
            const char* v = std::getenv("NSOS_D2H_TIMING");
            return v != nullptr && v[0] == '1';
        }();
        std::chrono::steady_clock::time_point d2h_started;
        if (d2h_timing) d2h_started = std::chrono::steady_clock::now();
        status = cudaMemcpy(
            h_result_.get(), d_result_.get(), sizeof(int),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            return poison_runtime(status, "greedy-token download");
        }
        record_gpu_transfer(
            Device::CPU, Device::GPU, sizeof(int));
        if (d2h_timing) {
            d2h_us_total_ += static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - d2h_started)
                    .count()) /
                1000.0;
            ++d2h_copies_;
        }
        const int token = *h_result_.get();
        // Commit incremental sampler state only after the synchronous D2H
        // boundary proves all preceding default-stream work completed.
        marked_count_ = generated_so_far;
        // -1 sentinel: every candidate was masked.  Fall back to the host greedy
        // branch, which scans for the first allowed token then EOS/0 (the GPU
        // kernel cannot reproduce that scan), so behaviour matches exactly.
        if (token < 0 || token >= vocab) return false;
        out_token = token;
        return true;
    }

    bool consume_failure(std::string& reason) noexcept {
        if (!poisoned_ || failure_reported_) return false;
        failure_reported_ = true;
        try {
            reason = disabled_reason_.empty()
                         ? "GPU sampler entered an unavailable state"
                         : disabled_reason_;
        } catch (...) {
            reason.clear();
        }
        return true;
    }

private:
    bool ensure(int vocab) {
        if (poisoned_) return false;
        if (vocab_ == vocab && d_seen_.get() != nullptr) return true;
        release_all();
        if (d_repeated_.get() != nullptr || d_seen_.get() != nullptr ||
            d_control_.get() != nullptr || d_result_.get() != nullptr ||
            h_result_.get() != nullptr) {
            return poison_without_runtime_error(
                "prior sampler allocation could not be released");
        }
        const size_t bytes = static_cast<size_t>(vocab);
        if (d_repeated_.ensure(bytes) == nullptr ||
            d_seen_.ensure(bytes) == nullptr ||
            d_control_.ensure(bytes) == nullptr ||
            d_result_.ensure(1) == nullptr ||
            h_result_.ensure(1) == nullptr) {
            (void)cudaGetLastError();
            release_all();
            if (d_repeated_.get() != nullptr || d_seen_.get() != nullptr ||
                d_control_.get() != nullptr || d_result_.get() != nullptr ||
                h_result_.get() != nullptr) {
                abandon_all();
            }
            poisoned_ = true;
            store_disabled_reason("sampler staging allocation failed");
            return false;
        }
        cudaError_t status =
            cudaMemset(d_repeated_.get(), 0, bytes);
        if (status == cudaSuccess) {
            status = cudaMemset(d_control_.get(), 0, bytes);
        }
        if (status != cudaSuccess) {
            return poison_runtime(status, "sampler-mask initialization");
        }
        vocab_ = vocab;
        control_built_ = false;
        marked_count_ = 0;
        return true;
    }
    bool build_control(int vocab,
                       const std::function<bool(int, int)>& is_control_token) {
        std::vector<unsigned char> host(static_cast<size_t>(vocab), 0);
        for (int t = 0; t < vocab; ++t) {
            host[static_cast<size_t>(t)] = is_control_token(t, vocab) ? 1 : 0;
        }
        const cudaError_t status =
            cudaMemcpy(d_control_.get(), host.data(),
                       static_cast<size_t>(vocab), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            return poison_runtime(status, "control-mask upload");
        }
        record_gpu_transfer(
            Device::GPU, Device::CPU, static_cast<size_t>(vocab));
        control_built_ = true;
        return true;
    }
    bool poison_runtime(cudaError_t status, const char* operation) noexcept {
        poisoned_ = true;
        abandon_all();
        try {
            disabled_reason_ =
                std::string(operation) + " failed on " +
                NSOS_GPU_BACKEND_NAME + ": " + cudaGetErrorString(status);
        } catch (...) {
            disabled_reason_.clear();
        }
        (void)cudaGetLastError();
        return false;
    }
    bool poison_without_runtime_error(const char* reason) noexcept {
        poisoned_ = true;
        abandon_all();
        store_disabled_reason(reason);
        return false;
    }
    void store_disabled_reason(const char* reason) noexcept {
        try {
            disabled_reason_ = reason != nullptr ? reason : "";
        } catch (...) {
            disabled_reason_.clear();
        }
    }
    void release_all() noexcept {
        d_repeated_.release();
        d_seen_.release();
        d_control_.release();
        d_result_.release();
        h_result_.release();
        vocab_ = 0;
        control_built_ = false;
        marked_count_ = 0;
    }
    void abandon_all() noexcept {
        d_repeated_.abandon();
        d_seen_.abandon();
        d_control_.abandon();
        d_result_.abandon();
        h_result_.abandon();
    }
    cuda_detail::DeviceBuffer<unsigned char> d_repeated_;
    cuda_detail::DeviceBuffer<unsigned char> d_seen_;
    cuda_detail::DeviceBuffer<unsigned char> d_control_;
    cuda_detail::DeviceBuffer<int> d_result_;
    cuda_detail::PinnedHostBuffer<int> h_result_;
    int vocab_ = 0;
    bool control_built_ = false;
    bool poisoned_ = false;
    bool failure_reported_ = false;
    size_t marked_count_ = 0;
    std::vector<int> banned_;
    std::string disabled_reason_;
    double d2h_us_total_ = 0.0;  // NSOS_D2H_TIMING accumulators
    size_t d2h_copies_ = 0;
};
#endif  // USE_CUDA

} // namespace

bool InferenceEngine::try_load_model_pack(const std::string& path,
                                          const ModelLoadOptions& options) {
    namespace fs = std::filesystem;

    fs::path pack_root(path);
    if (fs::is_regular_file(pack_root)) {
        pack_root = pack_root.parent_path();
    }
    const fs::path manifest_path = pack_root / "manifest.nsos";
    if (!fs::exists(manifest_path)) {
        return false;
    }

    const auto manifest = read_key_value_file(manifest_path);
    const auto format_it = manifest.find("format");
    const auto version_it = manifest.find("version");
    if (format_it == manifest.end() || format_it->second != "nsos-pack-v2" ||
        version_it == manifest.end() || version_it->second != "2") {
        throw std::runtime_error("Model pack manifest has unsupported format/version");
    }
    const auto get_value = [&](const std::string& key) -> std::string {
        const auto it = manifest.find(key);
        if (it == manifest.end()) {
            throw std::runtime_error("Model pack manifest missing key: " + key);
        }
        return it->second;
    };

    const fs::path config_path = pack_child_path(pack_root, get_value("config"), "config");
    const fs::path tokenizer_path = pack_child_path(pack_root, get_value("tokenizer"), "tokenizer");
    const fs::path weights_path = pack_child_path(pack_root, get_value("weights"), "weights");
    const bool has_edge_linear = manifest.count("edge_linear") > 0;
    bool quantization_ready = false;
    if (const auto ready_it = manifest.find("quantization_ready");
        ready_it != manifest.end()) {
        if (ready_it->second == "1" || ready_it->second == "true") {
            quantization_ready = true;
        } else if (ready_it->second != "0" && ready_it->second != "false") {
            throw std::runtime_error(
                "Model pack manifest has an invalid quantization_ready value");
        }
    }
    const fs::path edge_linear_path =
        has_edge_linear ? pack_child_path(pack_root, get_value("edge_linear"), "edge_linear")
                        : fs::path{};

    ensure_regular_file_within_limit(config_path, kMaxConfigFileBytes, "config");
    ensure_regular_file_within_limit(tokenizer_path, kMaxTokenizerPackBytes, "tokenizer");
    ensure_regular_file_within_limit(weights_path, kMaxWeightsPackBytes, "weights");
    if (has_edge_linear) {
        ensure_regular_file_within_limit(edge_linear_path, kMaxEdgePackBytes, "edge linear");
    }

    (void)require_pack_sha256(
        manifest, "config", "sha256_config");
    (void)require_pack_sha256(
        manifest, "tokenizer", "sha256_tokenizer");
    (void)require_pack_sha256(
        manifest, "weights", "sha256_weights");
    if (has_edge_linear) {
        (void)require_pack_sha256(
            manifest, "edge linear", "sha256_edge_linear");
    }
    verify_pack_file_digest(
        manifest, config_path, "config", "sha256_config");
    verify_pack_file_digest(
        manifest, tokenizer_path, "tokenizer", "sha256_tokenizer");

    ModelConfig pack_config = read_model_config(config_path);
    apply_model_load_options(pack_config, options);
    validate_model_config_for_pack(pack_config);
    this->config = pack_config;

    const Device device = this->config.use_cuda ? Device::GPU : Device::CPU;
    this->model = std::make_unique<JambaModel>(this->config, device);
    this->trainer = std::make_unique<Trainer>(this->model.get(), 0.001f);
    this->trainer->logit_l2_beta = this->config.logit_l2_beta;
    this->trainer->pantheon_vib_beta = this->config.pantheon_vib_beta;
    this->model->set_training_mode(false);

    this->tokenizer = Tokenizer();
    this->tokenizer.load(tokenizer_path.string());
    this->model->load(weights_path.string());
    if (has_edge_linear) {
        // A packed payload alone does not prove that the model was trained in
        // QAT. Post-training ternarization can cause a severe quality cliff.
        // Release FP32 automatically only when the producer recorded an
        // active quantized phase. Audits may force packed mode explicitly.
        const bool keep_fp32 =
            environment_flag("NSOS_KEEP_FP32_WEIGHTS");
        const bool force_packed =
            environment_flag("NSOS_FORCE_PACKED_WEIGHTS");
        if (keep_fp32 && force_packed) {
            throw std::runtime_error(
                "NSOS_KEEP_FP32_WEIGHTS and NSOS_FORCE_PACKED_WEIGHTS "
                "cannot both be enabled");
        }
        const bool release_fp32 =
            !keep_fp32 && (quantization_ready || force_packed);
        this->model->load_edge_linear_pack(edge_linear_path.string(), release_fp32);
        this->model->set_gpu_packed_inference(
            device == Device::GPU && release_fp32);
        if (release_fp32) {
            std::cerr << "[InferenceEngine] edge pack loaded; FP32 linear weights "
                      << "released (1.58-bit inference mode)."
                      << std::endl;
        } else {
            std::cerr << "[InferenceEngine] edge pack loaded; FP32 weights retained "
                      << "(reference mode: pack is not QAT-ready or retention "
                      << "was explicitly requested)."
                      << std::endl;
        }
    }
    // Weights and edge packs have their own structural integrity trailers.
    // Bind the successfully parsed artifacts back to the signed manifest
    // after loading so a replacement between path resolution and parsing
    // cannot silently publish a different model. Recheck the smaller config
    // and tokenizer for the same TOCTOU class.
    verify_pack_file_digest(
        manifest, config_path, "config", "sha256_config");
    verify_pack_file_digest(
        manifest, tokenizer_path, "tokenizer", "sha256_tokenizer");
    verify_pack_file_digest(
        manifest, weights_path, "weights", "sha256_weights");
    if (has_edge_linear) {
        verify_pack_file_digest(
            manifest, edge_linear_path, "edge linear",
            "sha256_edge_linear");
    }
    this->loaded_from_pack_ = true;
    return true;
}

InferenceEngine& InferenceEngine::operator=(InferenceEngine&& other) noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Tokenizer>);
    static_assert(std::is_nothrow_move_assignable_v<ModelConfig>);
    static_assert(std::is_nothrow_move_assignable_v<GenerationMetrics>);
    if (this == &other) {
        return *this;
    }

    // Trainer stores a non-owning model pointer. Destroy the destination
    // trainer before its model, then publish the source model before its
    // matching trainer so that relationship is valid throughout the move.
    trainer.reset();
    model.reset();
    model = std::move(other.model);
    trainer = std::move(other.trainer);
    tokenizer = std::move(other.tokenizer);
    config = std::move(other.config);
    last_metrics_ = std::move(other.last_metrics_);
    loaded_from_pack_ = other.loaded_from_pack_;
    inference_state_poisoned_.store(
        other.inference_state_poisoned_.load(std::memory_order_acquire),
        std::memory_order_release);
    other.loaded_from_pack_ = false;
    other.inference_state_poisoned_.store(false, std::memory_order_release);
    return *this;
}

void InferenceEngine::commit_loaded_engine(InferenceEngine&& staged) {
    this->model = std::move(staged.model);
    this->trainer = std::move(staged.trainer);
    this->tokenizer = std::move(staged.tokenizer);
    this->config = staged.config;
    this->last_metrics_ = {};
    this->loaded_from_pack_ = staged.loaded_from_pack_;
    this->inference_state_poisoned_.store(
        false, std::memory_order_release);
}

bool InferenceEngine::load_model_raw(const std::string& path,
                                     const ModelConfig& raw_config) {
    try {
        if (!path.empty()) {
            std::error_code exists_ec;
            if (!std::filesystem::exists(path, exists_ec) || exists_ec ||
                !std::filesystem::is_regular_file(path, exists_ec) || exists_ec) {
                std::cerr << "[InferenceEngine] Checkpoint not found or not a regular file at '"
                          << path << "'." << std::endl;
                return false;
            }
        }

        InferenceEngine staged;
        staged.config = raw_config;
        validate_model_config(staged.config);
        const Device device = staged.config.use_cuda ? Device::GPU : Device::CPU;
        staged.model = std::make_unique<JambaModel>(staged.config, device);
        staged.trainer = std::make_unique<Trainer>(staged.model.get(), 0.001f);
        staged.trainer->logit_l2_beta = staged.config.logit_l2_beta;
        staged.trainer->pantheon_vib_beta = staged.config.pantheon_vib_beta;
        staged.model->set_training_mode(false);
        staged.tokenizer = Tokenizer();
        staged.try_load_tokenizer(path);
        if (!path.empty()) {
            staged.model->load(path);
        }
        staged.loaded_from_pack_ = false;
        commit_loaded_engine(std::move(staged));
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Failed to load model from '" << path
                  << "': " << ex.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::load_model(const std::string& path) {
    try {
        if (!path.empty()) {
            InferenceEngine staged_pack;
            if (staged_pack.try_load_model_pack(path, ModelLoadOptions{})) {
                commit_loaded_engine(std::move(staged_pack));
                return true;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Failed to load model pack from '"
                  << path << "': " << ex.what() << std::endl;
        return false;
    }
    return load_model_raw(path, ModelConfig{});
}

bool InferenceEngine::load_model(const std::string& path,
                                 const ModelLoadOptions& options) {
    try {
        if (!path.empty()) {
            InferenceEngine staged_pack;
            if (staged_pack.try_load_model_pack(path, options)) {
                commit_loaded_engine(std::move(staged_pack));
                return true;
            }
        }
        std::cerr
            << "[InferenceEngine] ModelLoadOptions require an NSOS model pack; "
               "raw checkpoints need a complete ModelConfig."
            << std::endl;
        return false;
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Failed to load model pack from '"
                  << path << "': " << ex.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::load_model(const std::string& path,
                                 const ModelConfig& config_value) {
    try {
        if (!path.empty()) {
            // Compatibility overload: a supplied ModelConfig makes every
            // operational field below explicit, including false/default
            // values. Architecture and training-policy fields remain exactly
            // those signed by the pack configuration/checkpoint digest.
            ModelLoadOptions options;
            options.use_cuda = config_value.use_cuda;
            options.default_batch_size = config_value.default_batch_size;
            options.mcts_simulations = config_value.mcts_simulations;
            options.mcts_depth = config_value.mcts_depth;
            options.checkpoint_path = config_value.checkpoint_path;
            InferenceEngine staged_pack;
            if (staged_pack.try_load_model_pack(path, options)) {
                commit_loaded_engine(std::move(staged_pack));
                return true;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Failed to load model pack from '"
                  << path << "': " << ex.what() << std::endl;
        return false;
    }
    return load_model_raw(path, config_value);
}

void InferenceEngine::try_load_tokenizer(const std::string& path) {
    namespace fs = std::filesystem;

    if (path.empty()) {
        return;
    }

    std::vector<fs::path> candidates;
    if (!path.empty()) {
        const fs::path base(path);
        try {
            if (fs::exists(base)) {
                if (fs::is_regular_file(base) &&
                    (base.extension() == ".ox3" || base.extension() == ".txt" ||
                     base.extension() == ".tok" || base.filename() == "tokenizer.nsos")) {
                    candidates.push_back(base);
                }
                if (fs::is_directory(base)) {
                    candidates.push_back(base / "tokenizer.nsos");
                    candidates.push_back(base / "tokenizer.tok");
                    candidates.push_back(base / "tokenizer.ox3");
                    candidates.push_back(base / "tokenizer.txt");
                    candidates.push_back(base / "vocab.txt");
                    candidates.push_back(base / "bpe.txt");
                } else if (base.has_parent_path()) {
                    const fs::path parent = base.parent_path();
                    candidates.push_back(parent / "tokenizer.nsos");
                    candidates.push_back(parent / "tokenizer.tok");
                    candidates.push_back(parent / "tokenizer.ox3");
                    candidates.push_back(parent / "tokenizer.txt");
                    candidates.push_back(parent / "vocab.txt");
                    candidates.push_back(parent / "bpe.txt");
                }
            }
        } catch (const std::exception& ex) {
            throw std::runtime_error(std::string("Failed to inspect tokenizer candidates near '") +
                                     path + "': " + ex.what());
        }
    }

    std::vector<std::string> seen_candidates;
    std::vector<fs::path> deduped_candidates;
    deduped_candidates.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const std::string text = candidate.lexically_normal().string();
        if (std::find(seen_candidates.begin(), seen_candidates.end(), text) == seen_candidates.end()) {
            seen_candidates.push_back(text);
            deduped_candidates.push_back(candidate);
        }
    }

    std::vector<std::string> load_errors;
    bool found_file = false;
    for (const auto& candidate : deduped_candidates) {
        try {
            if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
                found_file = true;
                tokenizer.load(candidate.string());
                return;
            }
        } catch (const std::exception& ex) {
            load_errors.push_back(candidate.string() + ": " + ex.what());
        }
    }

    std::ostringstream message;
    if (found_file) {
        message << "Failed to load tokenizer from candidates near '" << path << "'";
        if (!load_errors.empty()) {
            message << " (";
            for (size_t i = 0; i < load_errors.size(); ++i) {
                if (i > 0) message << "; ";
                message << load_errors[i];
            }
            message << ")";
        }
    } else {
        message << "No tokenizer file found near '" << path << "'";
    }
    throw std::runtime_error(message.str());
}

std::vector<int> InferenceEngine::sanitize_token_ids(const std::vector<int>& ids) const {
    const int vocab = std::max(this->config.vocab_size, 2);
    for (int token : ids) {
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("Tokenizer/model vocabulary mismatch: token id " +
                                     std::to_string(token) + " is outside model vocab size " +
                                     std::to_string(vocab));
        }
    }
    return ids;
}

void InferenceEngine::load_tokenizer(const std::string& path) {
    if (!model) throw std::runtime_error("No model loaded");
    std::lock_guard<std::recursive_mutex> lock(model->execution_mutex_);
    Tokenizer staged;
    staged.load(path);
    if (staged.vocab_size <= 0 || staged.vocab_size > config.vocab_size) {
        throw std::invalid_argument("Tokenizer vocabulary exceeds the loaded model vocabulary");
    }
    tokenizer = std::move(staged);
}

Tensor InferenceEngine::forward_logits(const std::vector<int>& ids) {
    if (!model || inference_state_poisoned()) {
        throw std::runtime_error("Evaluation requires a usable loaded model");
    }
    std::lock_guard<std::recursive_mutex> lock(model->execution_mutex_);
    model->set_training_mode(false);
    model->set_streaming_inference(false);
    model->reset_session();
    StreamingInferenceGuard guard{model.get(), &inference_state_poisoned_};
    Tensor logits = model->forward_ids(ids, nullptr);
    guard.restore();
    return logits;
}

std::vector<int> InferenceEngine::tokenize(const std::string& text) {
    return tokenizer.encode(text);
}

std::string InferenceEngine::detokenize(const std::vector<int>& ids) const {
    return tokenizer.decode(ids);
}

ModelConfig InferenceEngine::model_config() const {
    if (!model) throw std::runtime_error("No model loaded");
    return model->model_config();
}

size_t InferenceEngine::parameter_count() const {
    if (!model) return 0;
    size_t count = 0;
    for (auto* parameter : model->parameters()) {
        if (parameter) count += static_cast<size_t>(parameter->data.size);
    }
    return count;
}

std::string InferenceEngine::generate(const std::string& prompt, int max_tokens,
                                      float temperature) {
    GenerationOptions options;
    options.max_tokens = max_tokens;
    options.temperature = temperature;
    options.max_context_tokens = this->config.max_context_tokens;
    return generate(prompt, options);
}

std::string InferenceEngine::generate(const std::string& prompt,
                                      const GenerationOptions& options) {
    return generate_stream(prompt, options, {});
}

std::string InferenceEngine::generate_stream(
    const std::string& prompt,
    const GenerationOptions& options,
    const std::function<void(const std::string&)>& on_chunk) {
    if (!this->model) {
        throw std::runtime_error("No model loaded");
    }
    if (inference_state_poisoned()) {
        throw std::runtime_error(
            "InferenceEngine is fail-stop poisoned after streaming cleanup "
            "failed; reload the model before reuse");
    }
    validate_generation_options(options, this->config.vocab_size);
    constexpr size_t kMaxDirectPromptBytes = 16ull * 1024ull * 1024ull;
    if (prompt.size() > kMaxDirectPromptBytes) {
        throw std::invalid_argument("prompt exceeds the direct SDK byte limit");
    }

    auto started_at = std::chrono::steady_clock::now();
    std::vector<int> prompt_tokens = sanitize_token_ids(this->tokenizer.encode(prompt));
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(1);
    }

    last_metrics_ = {};
    last_metrics_.prompt_tokens_total = prompt_tokens.size();
    last_metrics_.loaded_from_pack = loaded_from_pack_;

    const int context_limit = std::min(config.max_context_tokens,
        options.max_context_tokens > 0 ? options.max_context_tokens : config.max_context_tokens);
    std::vector<int> output = prompt_tokens;
    if (static_cast<int>(output.size()) > context_limit) {
        output.erase(output.begin(), output.end() - context_limit);
    }
    const size_t prompt_tokens_used = output.size();
    if (options.max_tokens == 0) {
        last_metrics_.prompt_tokens_used = prompt_tokens_used;
        last_metrics_.batch_size = 1;
        last_metrics_.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started_at).count();
        return {};
    }
    this->model->record_audit_token_context(output,
                                            1,
                                            prompt_tokens.size(),
                                            prompt_tokens_used,
                                            context_limit,
                                            prompt_tokens_used < prompt_tokens.size());

    const int top_k =
        options.top_k > 0
            ? options.top_k
            : std::min(std::max(this->config.vocab_size / 8, 8),
                       std::max(this->config.vocab_size - 1, 1));

    const uint64_t sampler_sequence =
        fnv1a_hash_text(prompt) ^
        (static_cast<uint64_t>(std::max(options.max_tokens, 0)) << 32) ^
        static_cast<uint64_t>(std::max(options.eos_token_id, 0));
    std::mt19937 rng = make_sampler_rng("generate_stream", sampler_sequence);
    SamplerWorkspace sampler_workspace;
    std::vector<std::string> token_piece_cache;
    std::vector<char> token_piece_loaded;

    auto cached_token_piece = [&](int token, int vocab_size) -> const std::string& {
        if (token_piece_cache.size() != static_cast<size_t>(vocab_size)) {
            token_piece_cache.assign(static_cast<size_t>(vocab_size), std::string{});
            token_piece_loaded.assign(static_cast<size_t>(vocab_size), 0);
        }
        if (!token_piece_loaded[static_cast<size_t>(token)]) {
            token_piece_cache[static_cast<size_t>(token)] =
                decode_token_piece(this->tokenizer, token);
            token_piece_loaded[static_cast<size_t>(token)] = 1;
        }
        return token_piece_cache[static_cast<size_t>(token)];
    };

#ifdef USE_CUDA
    GpuGreedySampler gpu_greedy_sampler;
    // CUDA Graphs (opt-in NSOS_CUDA_GRAPH): one-time capability + self-test probe
    // on the target GPU.  Graph-capture validity is a GPU-runtime property, so we
    // verify capture==eager once before any graphed decode is relied upon.  Logged
    // once per process; never gates the hot path (decode still runs eagerly).
    {
        static std::once_flag cuda_graph_probe_once;
        std::call_once(cuda_graph_probe_once, [] {
            const char* graph_env = std::getenv("NSOS_CUDA_GRAPH");
            if (graph_env && graph_env[0] == '1') {
                const int supported = cuda_graphs_supported();
                const int self_test = supported ? cuda_graph_self_test() : 0;
                std::fprintf(
                    stderr,
                    "[nsos] CUDA Graphs probe: supported=%d self_test=%s\n",
                    supported, self_test ? "PASS" : "FAIL");
            }
        });
    }
#endif
    auto sample_next_token = [&](const Tensor& logits) -> int {
#ifdef USE_CUDA
        // On-device greedy decode (opt-in NSOS_GPU_SAMPLER): keep selection on
        // the GPU, skipping the per-token [vocab] D2H + host vocab scan.  Falls
        // back to the host path for stochastic sampling or on any CUDA error.
        if (gpu_greedy_sampler_enabled() && logits.get_device() == Device::GPU &&
            logits.size > 0) {
            const int gpu_vocab = logits.shape.back();
            const bool greedy = options.temperature <= 1e-5f || top_k == 1;
            const bool top_p_can_change =
                top_k != 1 && options.top_p < 1.0f && options.top_p > 0.0f;
            if (greedy && !top_p_can_change && gpu_vocab > 0) {
                const float* raw_row =
                    logits.raw_data() + (logits.size - gpu_vocab);
                int gpu_tok = options.eos_token_id;
                if (gpu_greedy_sampler.select(
                        raw_row, gpu_vocab, options, output, prompt_tokens_used,
                        [&](int token, int v) {
                            return cached_token_piece(token, v).rfind("<|", 0) ==
                                   0;
                        },
                        gpu_tok)) {
                    return gpu_tok;
                }
                std::string sampler_failure;
                if (gpu_greedy_sampler.consume_failure(sampler_failure)) {
                    if (sampler_failure.empty()) {
                        sampler_failure =
                            "GPU sampler entered an unavailable state";
                    }
                    if (strict_gpu_execution()) {
                        throw std::runtime_error(sampler_failure);
                    }
                    std::fprintf(
                        stderr,
                        "[nsos] GPU sampler disabled for this generation; "
                        "using audited host sampling fallback: %s\n",
                        sampler_failure.c_str());
                }
            }
        }
#endif
        if (logits.shape.empty()) {
            throw std::runtime_error("Model returned logits without a vocabulary dimension");
        }
        const int columns = logits.shape.back();
        Tensor last_row = logits;
        if (columns > 0 && logits.size > columns) {
            const int rows = logits.size / columns;
            last_row = logits.reshape({rows, columns}).slice(0, rows - 1, rows);
        }
        Tensor host_logits = last_row.get_device() == Device::GPU ? last_row.cpu() : last_row;
        if (host_logits.size == 0) {
            return options.eos_token_id;
        }
        const int vocab_size = host_logits.shape.back();
        if (vocab_size <= 0 || vocab_size > host_logits.size) {
            throw std::runtime_error("Model returned an invalid logits shape");
        }
        const int last_offset = host_logits.size - vocab_size;
        return sample_from_host_logits_row(
            host_logits.data() + last_offset,
            vocab_size,
            top_k,
            options,
            output,
            prompt_tokens_used,
            rng,
            sampler_workspace,
            [&](int token, int current_vocab_size) {
                return cached_token_piece(token, current_vocab_size).rfind("<|", 0) == 0;
            },
            &last_metrics_.sampler_ms);
    };

    this->model->set_training_mode(false);
    this->model->reset_runtime_telemetry();
    const bool can_use_streaming = this->model->supports_streaming_inference();
    last_metrics_.used_streaming = can_use_streaming;
    this->model->reset_session();
    this->model->set_streaming_inference(can_use_streaming);
    StreamingInferenceGuard streaming_guard{
        this->model.get(), &inference_state_poisoned_};

    // ─────────────────────────────────────────────────────────────────
    // INFERENCE BOTTLENECK #1 mitigation (2026-05-17):
    // If the model reports !supports_streaming_inference, the decode
    // loop falls into the O(N²) re-process-everything path below.  For
    // a 50-token prompt + 200 generated tokens that's ~30,000 tokens
    // re-processed across the run — and the per-step time grows
    // quadratically.  v10 inference observed slow decode that the user
    // attributed to baseline cost; this path being silently selected
    // is a plausible additional contributor.
    //
    // We emit a one-shot stderr warning the first time we hit the
    // fallback so deployments don't silently degrade.  Without this
    // warning, telemetry has no signal that the slow path was taken
    // (used_streaming is set in last_metrics_ but production users
    // rarely check it).
    if (!can_use_streaming) {
        static std::once_flag warned_once;
        std::call_once(warned_once, [] {
            std::cerr << "[nsos][WARN] decode falling back to O(N^2) "
                         "re-process path: model->supports_streaming_inference() "
                         "returned false.  Every generated token re-processes "
                         "the entire output history, so long generations "
                         "become exponentially slower per token.  "
                         "Investigate which layer type returns false from "
                         "the streaming check (see JambaModel::"
                         "supports_streaming_inference)."
                      << std::endl;
        });
    }

    // ─────────────────────────────────────────────────────────────────
    // INFERENCE BOTTLENECK #2 mitigation (2026-05-17):
    // Pre-allocate the KV cache to the FULL expected sequence length
    // before the decode loop runs.  The default cache page size is 64
    // tokens; without pre-allocation, the cache grows in pages,
    // triggering a realloc + memcpy of the entire cache content every
    // 64 generated tokens.  For 512 tokens that's 8 reallocs each
    // copying an ever-growing block (last realloc copies ~448 tokens
    // worth of KV state for every attention layer).  Pre-allocating
    // is one ensure_kv_cache_capacity call per attention layer.
    //
    // The reserve target is prompt + requested new tokens.  If the
    // user's prompt was truncated to fit context_limit, `output`
    // already reflects the truncated length so reserving for
    // output.size() + max_tokens is correct.
    {
        const int reserve_total = static_cast<int>(output.size()) +
                                   std::max(options.max_tokens, 0);
        if (reserve_total > 0) {
            // Device: take it from the model's first parameter. Inference
            // engines set device placement consistently at load time.
            try {
                const Device device = this->model->parameters().empty()
                    ? Device::CPU
                    : this->model->parameters().front()->data.get_device();
                this->model->reserve_kv_cache(reserve_total, device, 1);
            } catch (const std::exception& error) {
                // Fail closed: silent incremental growth hides OOM/backend
                // defects and changes latency within the same generation.
                throw std::runtime_error(
                    std::string("KV-cache reservation failed before decode: ") +
                    error.what());
            }
        }
    }

    auto prefill_started_at = std::chrono::steady_clock::now();
    auto decode_started_at = prefill_started_at;
    auto decode_finished_at = prefill_started_at;
    if (can_use_streaming) {
        Tensor logits;
        if (!output.empty()) {
            logits = this->model->forward_ids_last(output, nullptr);
        }
        decode_started_at = std::chrono::steady_clock::now();

        // HIP/CUDA graph decode (opt-in NSOS_GPU_GRAPH_DECODE=1):
        // the model captures one single-token forward on its explicit stream
        // and replays it per token.  An empty return means the graph path is
        // unavailable/disabled — the SAME token then runs through the eager
        // forward_ids, so results are identical either way.
        bool try_decode_graph = true;
        for (int step = 0; step < std::max(options.max_tokens, 0); ++step) {
            const int next_token = sample_next_token(logits);
            output.push_back(next_token);

            if (next_token == options.eos_token_id) {
                break;
            }
            const std::string piece = decode_token_piece(this->tokenizer, next_token);
            if (on_chunk) {
                on_chunk(piece);
            }

            if (step + 1 == options.max_tokens) break;

            if (try_decode_graph) {
                Tensor graphed = this->model->forward_ids_decode_graph(next_token);
                if (graphed.size > 0) {
                    logits = std::move(graphed);
                    continue;
                }
                try_decode_graph = false;
            }
            logits = this->model->forward_ids_last({next_token}, nullptr);
        }
        decode_finished_at = std::chrono::steady_clock::now();
    } else {
        decode_started_at = std::chrono::steady_clock::now();
        for (int step = 0; step < std::max(options.max_tokens, 0); ++step) {
            std::vector<int> model_input = output;
            if (static_cast<int>(model_input.size()) > context_limit) {
                model_input.erase(model_input.begin(), model_input.end() - context_limit);
            }

            this->model->reset_session();
            Tensor logits = this->model->forward_ids_last(model_input, nullptr);
            const int next_token = sample_next_token(logits);
            output.push_back(next_token);

            if (next_token == options.eos_token_id) {
                break;
            }
            const std::string piece = decode_token_piece(this->tokenizer, next_token);
            if (on_chunk) {
                on_chunk(piece);
            }
        }
        decode_finished_at = std::chrono::steady_clock::now();
    }
    this->model->set_streaming_inference(false);

    std::string result;
    for (size_t i = prompt_tokens_used; i < output.size(); ++i) {
        if (output[i] == options.eos_token_id) {
            continue;
        }
        result += decode_token_piece(this->tokenizer, output[i]);
    }

    const auto finished_at = std::chrono::steady_clock::now();
    last_metrics_.prompt_tokens_used = prompt_tokens_used;
    last_metrics_.generated_tokens = output.size() - prompt_tokens_used;
    last_metrics_.batch_size = 1;
    last_metrics_.elapsed_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(finished_at - started_at)
                .count());
    last_metrics_.elapsed_ms /= 1000.0;
    last_metrics_.prefill_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(decode_started_at -
                                                                  prefill_started_at)
                .count()) /
        1000.0;
    last_metrics_.decode_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(decode_finished_at -
                                                                  decode_started_at)
                .count()) /
        1000.0;
    const double elapsed_seconds = std::max(last_metrics_.elapsed_ms / 1000.0, 1e-9);
    const double prefill_seconds = std::max(last_metrics_.prefill_ms / 1000.0, 1e-9);
    const double decode_seconds = std::max(last_metrics_.decode_ms / 1000.0, 1e-9);
    last_metrics_.prompt_tokens_per_sec =
        static_cast<double>(last_metrics_.prompt_tokens_used) /
        (can_use_streaming ? prefill_seconds : elapsed_seconds);
    last_metrics_.decode_tokens_per_sec =
        static_cast<double>(last_metrics_.generated_tokens) /
        (last_metrics_.decode_ms > 0.0 ? decode_seconds : elapsed_seconds);
    last_metrics_.total_tokens_per_sec =
        static_cast<double>(last_metrics_.prompt_tokens_used + last_metrics_.generated_tokens) /
        elapsed_seconds;
    const RuntimeTelemetrySnapshot runtime = this->model->runtime_telemetry();
    last_metrics_.mamba_fast_path_hits = runtime.mamba_fast_path_hits;
    last_metrics_.mamba_fast_path_fallbacks = runtime.mamba_fast_path_fallbacks;
    last_metrics_.mamba_stream_priming_gpu_calls =
        runtime.stream_priming_gpu_calls;
    last_metrics_.mamba_stream_priming_host_fallbacks =
        runtime.stream_priming_host_fallbacks;
    last_metrics_.mamba_last_fallback_reason = runtime.mamba_last_fallback_reason;
    streaming_guard.restore();
    return result;
}

std::vector<std::string> InferenceEngine::generate_batch(
    const std::vector<std::string>& prompts, const GenerationOptions& options) {
    if (!model || inference_state_poisoned()) {
        throw std::runtime_error("Batch generation requires a usable loaded model");
    }
    validate_generation_options(options, config.vocab_size);
    if (prompts.size() > 1024) throw std::invalid_argument("prompt batch exceeds the direct SDK item limit");
    size_t bytes = 0;
    for (const auto& prompt : prompts) {
        if (prompt.size() > 64ull * 1024ull * 1024ull - bytes)
            throw std::invalid_argument("prompt batch exceeds the direct SDK byte limit");
        bytes += prompt.size();
    }
    const auto started = std::chrono::steady_clock::now();
    auto milliseconds = [](auto start) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };
    GenerationMetrics metrics{};
    metrics.batch_size = prompts.size();
    metrics.loaded_from_pack = loaded_from_pack_;
    std::vector<std::string> result(prompts.size());
    if (prompts.empty()) { last_metrics_ = metrics; return result; }
    const int context_limit = std::min(config.max_context_tokens,
        options.max_context_tokens > 0 ? options.max_context_tokens : config.max_context_tokens);
    const int top_k = options.top_k > 0 ? options.top_k :
        std::min(std::max(config.vocab_size / 8, 8), std::max(config.vocab_size - 1, 1));
    struct Item {
        std::vector<int> tokens;
        size_t prompt_size = 0;
        bool finished = false;
        Tensor logits;
        JambaSessionSnapshot snapshot;
        std::mt19937 rng;
        SamplerWorkspace sampler;
#ifdef USE_CUDA
        std::unique_ptr<GpuGreedySampler> gpu_sampler;
#endif
    };
    std::vector<Item> items(prompts.size());
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t index = 0; index < prompts.size(); ++index) {
        auto& item = items[index];
        item.tokens = sanitize_token_ids(tokenizer.encode(prompts[index]));
        if (item.tokens.empty()) item.tokens.push_back(1);
        metrics.prompt_tokens_total += item.tokens.size();
        if (item.tokens.size() > static_cast<size_t>(context_limit))
            item.tokens.erase(item.tokens.begin(), item.tokens.end() - context_limit);
        item.prompt_size = item.tokens.size();
        metrics.prompt_tokens_used += item.prompt_size;
        const uint64_t sequence = fnv1a_hash_text(prompts[index]) ^
            (static_cast<uint64_t>(options.max_tokens) << 32) ^
            static_cast<uint64_t>(std::max(options.eos_token_id, 0));
        item.rng = make_sampler_rng("generate_stream", sequence);
        groups[item.prompt_size].push_back(index);
#ifdef USE_CUDA
        item.gpu_sampler = std::make_unique<GpuGreedySampler>();
#endif
    }
    std::vector<int8_t> controls(static_cast<size_t>(config.vocab_size), -1);
    auto control = [&](int token, int) {
        auto& cached = controls.at(static_cast<size_t>(token));
        if (cached < 0) cached = decode_token_piece(tokenizer, token).rfind("<|", 0) == 0 ? 1 : 0;
        return cached == 1;
    };
    auto sample = [&](Item& item) {
        const Tensor& logits = item.logits;
        const int vocab = logits.shape.back();
#ifdef USE_CUDA
        const bool greedy = options.temperature <= 1e-5f || top_k == 1;
        const bool nucleus = top_k != 1 && options.top_p < 1.0f && options.top_p > 0;
        if (gpu_greedy_sampler_enabled() && logits.get_device() == Device::GPU && greedy && !nucleus) {
            const auto start = std::chrono::steady_clock::now();
            int selected = options.eos_token_id;
            if (item.gpu_sampler->select(logits.raw_data() + logits.size - vocab,
                    vocab, options, item.tokens, item.prompt_size, control, selected)) {
                metrics.sampler_ms += milliseconds(start);
                return selected;
            }
            std::string failure;
            if (item.gpu_sampler->consume_failure(failure) && strict_gpu_execution())
                throw std::runtime_error(failure);
        }
#endif
        Tensor host = logits.get_device() == Device::GPU ? logits.cpu() : logits;
        return sample_from_host_logits_row(host.data() + host.size - vocab,
            vocab, top_k, options, item.tokens, item.prompt_size, item.rng,
            item.sampler, control, &metrics.sampler_ms);
    };
    model->set_training_mode(false);
    model->reset_runtime_telemetry();
    metrics.used_streaming = model->supports_streaming_inference();
    StreamingInferenceGuard guard{model.get(), &inference_state_poisoned_};

    // Equal-length buckets preserve each attention position. Different prompt
    // lengths no longer force the entire request into serial per-token decode.
    for (const auto& entry : groups) {
        const auto& indices = entry.second;
        if (options.max_tokens == 0) continue;
        const auto prefill_start = std::chrono::steady_clock::now();
        std::vector<JambaSessionSnapshot> snapshots;
        snapshots.reserve(indices.size());
        for (size_t index : indices) {
            auto& item = items[index];
            model->reset_session();
            model->set_streaming_inference(true);
            item.logits = model->forward_ids_last(item.tokens, nullptr);
            item.snapshot = model->fork_session(true);
            snapshots.push_back(item.snapshot);
        }
        const bool batched = model->supports_batched_streaming_inference();
        if (batched) {
            model->restore_session_batch(snapshots);
            const Device device = items[indices.front()].logits.get_device();
            model->reserve_kv_cache(static_cast<int>(entry.first) + options.max_tokens,
                                    device, static_cast<int>(indices.size()));
            snapshots.clear();
            for (size_t index : indices) items[index].snapshot = {};
        }
        metrics.prefill_ms += milliseconds(prefill_start);
        const auto decode_start = std::chrono::steady_clock::now();
        for (int step = 0; step < options.max_tokens; ++step) {
            bool active = false;
            std::vector<std::vector<int>> next(indices.size(), std::vector<int>(1, 0));
            for (size_t row = 0; row < indices.size(); ++row) {
                auto& item = items[indices[row]];
                if (item.finished) continue;
                const int token = sample(item);
                item.tokens.push_back(token);
                ++metrics.generated_tokens;
                next[row][0] = token;
                item.finished = token == options.eos_token_id;
                active = active || !item.finished;
            }
            if (!active || step + 1 == options.max_tokens) break;
            if (batched) {
                // Stable row slots avoid snapshot/download/restore every token.
                // Finished rows are independent and cannot affect live sessions.
                Tensor logits = model->forward_ids_batch(next, nullptr);
                const int vocab = logits.shape.back();
                Tensor rows = logits.reshape({static_cast<int>(indices.size()), vocab});
                for (size_t row = 0; row < indices.size(); ++row) {
                    if (!items[indices[row]].finished)
                        items[indices[row]].logits = rows.slice(0, static_cast<int>(row), static_cast<int>(row) + 1);
                }
            } else {
                for (size_t row = 0; row < indices.size(); ++row) {
                    auto& item = items[indices[row]];
                    if (item.finished) continue;
                    model->restore_session(item.snapshot);
                    item.logits = model->forward_ids_last(next[row], nullptr);
                    item.snapshot = model->fork_session(true);
                }
            }
        }
        metrics.decode_ms += milliseconds(decode_start);
    }
    for (size_t index = 0; index < items.size(); ++index) {
        const auto& item = items[index];
        for (size_t i = item.prompt_size; i < item.tokens.size(); ++i)
            if (item.tokens[i] != options.eos_token_id) result[index] += decode_token_piece(tokenizer, item.tokens[i]);
    }
    metrics.elapsed_ms = milliseconds(started);
    metrics.prompt_tokens_per_sec = metrics.prompt_tokens_used / std::max(metrics.prefill_ms / 1000.0, 1e-9);
    metrics.decode_tokens_per_sec = metrics.generated_tokens / std::max(metrics.decode_ms / 1000.0, 1e-9);
    metrics.total_tokens_per_sec = (metrics.prompt_tokens_used + metrics.generated_tokens) /
        std::max(metrics.elapsed_ms / 1000.0, 1e-9);
    const auto telemetry = model->runtime_telemetry();
    metrics.mamba_fast_path_hits = telemetry.mamba_fast_path_hits;
    metrics.mamba_fast_path_fallbacks = telemetry.mamba_fast_path_fallbacks;
    metrics.mamba_stream_priming_gpu_calls = telemetry.stream_priming_gpu_calls;
    metrics.mamba_stream_priming_host_fallbacks = telemetry.stream_priming_host_fallbacks;
    metrics.mamba_last_fallback_reason = telemetry.mamba_last_fallback_reason;
    last_metrics_ = metrics;
    guard.restore();
    return result;
}

float InferenceEngine::train_step(const std::vector<int>& input,
                                  const std::vector<int>& target) {
    if (inference_state_poisoned()) {
        throw std::runtime_error(
            "InferenceEngine is fail-stop poisoned after streaming cleanup "
            "failed; reload the model before training");
    }
    if (!this->model || !this->trainer || input.empty()) {
        return 0.0f;
    }

    std::vector<int> sanitized_input = sanitize_token_ids(input);
    std::vector<int> sanitized_target = sanitize_token_ids(target);
    if (sanitized_target.empty() && sanitized_input.size() < 2) {
        return 0.0f;
    }

    return this->trainer->train_step(sanitized_input, sanitized_target);
}

float InferenceEngine::train_step(const std::string& text) {
    std::vector<int> tokens = sanitize_token_ids(this->tokenizer.encode(text));
    if (tokens.size() < 2) {
        return 0.0f;
    }
    return train_step(tokens, {});
}

void InferenceEngine::request_training_cancellation() noexcept {
    if (trainer) {
        trainer->request_cancellation();
    }
}

void InferenceEngine::clear_training_cancellation() noexcept {
    if (trainer) {
        trainer->clear_cancellation();
    }
}

bool InferenceEngine::training_cancellation_requested() const noexcept {
    return trainer && trainer->cancellation_requested();
}

bool InferenceEngine::training_optimizer_state_poisoned() const noexcept {
    return trainer && trainer->optimizer_state_poisoned();
}

bool InferenceEngine::inference_state_poisoned() const noexcept {
    return inference_state_poisoned_.load(std::memory_order_acquire);
}

bool InferenceEngine::save_checkpoint(const std::string& path) const {
    if (!this->model || path.empty()) {
        return false;
    }
    const std::filesystem::path destination(path);
    const std::filesystem::path temporary = atomic_temp_path(destination);
    try {
        if (inference_state_poisoned()) {
            throw std::runtime_error(
                "InferenceEngine is fail-stop poisoned after streaming cleanup");
        }
        // Canonical lock ordering is Trainer -> model. This binds the poison
        // decision and serialized weights to the same training transaction.
        std::unique_lock<std::recursive_mutex> trainer_lock;
        if (this->trainer) {
            trainer_lock = std::unique_lock<std::recursive_mutex>(
                this->trainer->state_mutex_);
            this->trainer->ensure_optimizer_state_usable();
        }
        std::lock_guard<std::recursive_mutex> model_lock(
            this->model->execution_mutex_);
        if (!destination.parent_path().empty()) {
            std::filesystem::create_directories(destination.parent_path());
        }
        this->model->save(temporary.string());
        replace_file(temporary, destination);
        return true;
    } catch (const std::exception& ex) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        std::cerr << "[NSOS] Failed to save checkpoint '" << path << "': "
                  << ex.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::save_model_pack(const std::string& directory) const {
    namespace fs = std::filesystem;

    if (!this->model || directory.empty()) {
        return false;
    }

    const fs::path pack_root(directory);
    try {
        if (inference_state_poisoned()) {
            throw std::runtime_error(
                "InferenceEngine is fail-stop poisoned after streaming cleanup");
        }
        // QAT readiness and every artifact must describe one immutable
        // generation. Hold the canonical Trainer -> model lock order through
        // publication so another step cannot race the manifest metadata.
        std::unique_lock<std::recursive_mutex> trainer_lock;
        if (this->trainer) {
            trainer_lock = std::unique_lock<std::recursive_mutex>(
                this->trainer->state_mutex_);
            this->trainer->ensure_optimizer_state_usable();
        }
        std::lock_guard<std::recursive_mutex> model_lock(
            this->model->execution_mutex_);

        fs::create_directories(pack_root);

        const fs::path manifest_path = pack_root / "manifest.nsos";
        const std::string generation_id =
            atomic_temp_path(pack_root / "generation")
                .filename()
                .string();
        const fs::path generation_root =
            pack_root / "generations" / generation_id;
        fs::create_directories(generation_root);
        const fs::path weights_path =
            generation_root / "model.nsos.bin";
        const fs::path tokenizer_path =
            generation_root / "tokenizer.nsos";
        const fs::path config_path =
            generation_root / "config.nsos";
        const fs::path edge_linear_path =
            generation_root / "edge_linear.nsos";
        const auto manifest_child = [&](const fs::path& child) {
            const fs::path relative =
                child.lexically_relative(pack_root);
            if (relative.empty() || relative.is_absolute() ||
                path_has_parent_traversal(relative)) {
                throw std::logic_error(
                    "Generated model-pack child escaped its root");
            }
            return relative.generic_string();
        };
        const std::string weights_manifest_path =
            manifest_child(weights_path);
        const std::string edge_manifest_path =
            manifest_child(edge_linear_path);
        const std::string tokenizer_manifest_path =
            manifest_child(tokenizer_path);
        const std::string config_manifest_path =
            manifest_child(config_path);
        bool quantization_ready =
            this->trainer != nullptr &&
            this->trainer->progressive_qat_active();
        bool saw_quantizable_layer = false;
        for (const BitLinear* layer : this->model->collect_bitlinear_layers()) {
            if (!layer || layer->quantization_sensitive()) {
                continue;
            }
            saw_quantizable_layer = true;
            if (layer->reference_path_enabled()) {
                quantization_ready = false;
                break;
            }
        }
        quantization_ready = quantization_ready && saw_quantizable_layer;

        try {
            const ModelConfig exported_config =
                this->model->save_model_pack_artifacts(
                    weights_path.string(),
                    edge_linear_path.string());
            this->tokenizer.save_pack(tokenizer_path.string());
            sync_file_to_storage(tokenizer_path);
            write_model_config(config_path, exported_config);

            // Every generation is immutable and unreachable until this one
            // atomic manifest replacement. A crash cannot overwrite children
            // referenced by the previous manifest.
            write_key_value_file(
                manifest_path,
                {
                    {"format", "nsos-pack-v2"},
                    {"version", "2"},
                    {"quantization_ready", quantization_ready ? "1" : "0"},
                    {"weights", weights_manifest_path},
                    {"edge_linear", edge_manifest_path},
                    {"tokenizer", tokenizer_manifest_path},
                    {"config", config_manifest_path},
                    {"sha256_weights", sha256_checksum_file(weights_path)},
                    {"sha256_edge_linear", sha256_checksum_file(edge_linear_path)},
                    {"sha256_tokenizer", sha256_checksum_file(tokenizer_path)},
                    {"sha256_config", sha256_checksum_file(config_path)},
                });
        } catch (...) {
            // write_key_value_file can report a directory-flush failure after
            // the manifest rename itself committed. Never delete a generation
            // that the visible manifest already references.
            bool generation_is_published = false;
            try {
                if (fs::is_regular_file(manifest_path)) {
                    const auto visible_manifest =
                        read_key_value_file(manifest_path);
                    const auto visible_weights =
                        visible_manifest.find("weights");
                    generation_is_published =
                        visible_weights != visible_manifest.end() &&
                        visible_weights->second ==
                            weights_manifest_path;
                }
            } catch (...) {
                generation_is_published = false;
            }
            if (!generation_is_published) {
                std::error_code ignored;
                fs::remove_all(generation_root, ignored);
            }
            throw;
        }

        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[NSOS] Failed to save model pack '" << directory << "': "
                  << ex.what() << std::endl;
        return false;
    }
}

std::unique_ptr<InferenceEngine> InferenceEngine::clone_for_inference() const {
    if (!this->model) {
        throw std::runtime_error("Cannot clone inference engine without a loaded model");
    }
    if (inference_state_poisoned()) {
        throw std::runtime_error(
            "Cannot clone a fail-stop poisoned inference engine");
    }
    // A pre-lock atomic poison check can race a failing optimizer: the clone
    // would wait for the model lock and then copy the ambiguous post-failure
    // weights. Freeze Trainer first, validate poison under that transaction,
    // and only then freeze the model.
    std::unique_lock<std::recursive_mutex> trainer_lock;
    if (this->trainer) {
        trainer_lock = std::unique_lock<std::recursive_mutex>(
            this->trainer->state_mutex_);
        this->trainer->ensure_optimizer_state_usable();
    }
    std::lock_guard<std::recursive_mutex> model_lock(
        this->model->execution_mutex_);

    auto replica = std::make_unique<InferenceEngine>();
    replica->config = this->model->model_config();
    replica->loaded_from_pack_ = this->loaded_from_pack_;
    replica->tokenizer = this->tokenizer;

    const Device device = replica->config.use_cuda ? Device::GPU : Device::CPU;
    replica->model = std::make_unique<JambaModel>(replica->config, device);
    // Serving replicas are immutable and never execute admin training. Avoid a
    // Trainer object and optimizer-facing state in every replica.
    replica->trainer.reset();
    replica->model->set_training_mode(false);
    replica->model->set_streaming_inference(false);

    const auto source_parameters = this->model->parameters();
    const auto replica_parameters = replica->model->parameters();
    if (source_parameters.size() != replica_parameters.size()) {
        throw std::runtime_error("Inference replica clone parameter count mismatch");
    }
    for (size_t index = 0; index < source_parameters.size(); ++index) {
        Parameter* src = source_parameters[index];
        Parameter* dst = replica_parameters[index];
        if (src == nullptr || dst == nullptr) {
            throw std::runtime_error("Inference replica clone encountered null parameter");
        }
        // After an edge-pack load the source's packed BitLinear weights are
        // released (empty data) -- and with mixed precision only SOME layers
        // are released.  Released params are restored on the replica from the
        // packed state in the BitLinear loop below, so skip them here instead
        // of failing the shape check.
        if (src->data.size == 0) {
            continue;
        }
        if (src->base_name != dst->base_name || src->data.shape != dst->data.shape) {
            throw std::runtime_error("Inference replica clone parameter layout mismatch at index " +
                                     std::to_string(index));
        }
        dst->data.copy_from(src->data);
        dst->version = src->version;
    }

    const auto source_bitlinear_layers = this->model->collect_bitlinear_layers();
    const auto replica_bitlinear_layers = replica->model->collect_bitlinear_layers();
    if (source_bitlinear_layers.size() != replica_bitlinear_layers.size()) {
        throw std::runtime_error("Inference replica clone bitlinear layout mismatch");
    }
    for (size_t index = 0; index < source_bitlinear_layers.size(); ++index) {
        BitLinear* src = source_bitlinear_layers[index];
        BitLinear* dst = replica_bitlinear_layers[index];
        if (src == nullptr || dst == nullptr) {
            throw std::runtime_error("Inference replica clone encountered null bitlinear layer");
        }
        const BitLinearPackedState state = src->export_packed_state();
        const bool source_has_full_precision =
            src->has_full_precision_weight();
        auto prepared = dst->prepare_packed_state(
            state, device, !source_has_full_precision,
            source_has_full_precision ? &dst->weight.data : nullptr);
        dst->commit_prepared_packed_state(std::move(prepared));
        dst->set_reference_path(src->reference_path_enabled());
        dst->set_gpu_packed_inference(src->gpu_packed_inference_enabled());
    }
    // Importing packed caches must not mutate the model's declared parameter
    // alias topology (notably embedding.weight <-> value_head.weight).
    (void)replica->model->parameter_aliases();

    replica->last_metrics_ = this->last_metrics_;
    replica->model->set_training_rng_sequence(
        this->model->training_rng_sequence());
    return replica;
}

std::unique_ptr<InferenceEngine> InferenceEngine::clone_for_training() const {
    if (!this->model || !this->trainer) {
        throw std::runtime_error("Cannot clone training engine without a loaded trainer");
    }
    if (inference_state_poisoned()) {
        throw std::runtime_error(
            "Cannot clone a fail-stop poisoned inference engine");
    }
    std::lock_guard<std::recursive_mutex> state_lock(
        this->trainer->state_mutex_);
    std::lock_guard<std::recursive_mutex> model_lock(
        this->model->execution_mutex_);
    this->trainer->ensure_optimizer_state_usable();

    auto clone = clone_for_inference();
    clone->model->set_training_mode(true);
    clone->model->set_streaming_inference(false);
    clone->trainer = std::make_unique<Trainer>(clone->model.get(), this->trainer->learning_rate);

    const auto source_parameters = this->model->parameters();
    const auto target_parameters = clone->model->parameters();
    for (Parameter* parameter : source_parameters) {
        if (!parameter || parameter->data.size == 0) {
            throw std::runtime_error(
                "Training cannot start from an inference-only packed model");
        }
    }
    this->trainer->clone_runtime_state_to(
        *clone->trainer, source_parameters,
        target_parameters);
    return clone;
}

bool InferenceEngine::try_evaluate_simple_math(const std::string& prompt,
                                               const std::string& response,
                                               bool& is_valid) const {
    std::string expr = trim_copy(prompt);
    while (!expr.empty() && expr.back() == '=') {
        expr.pop_back();
    }
    expr = trim_copy(expr);

    const std::string ops = "+-*/";
    const std::size_t op_pos = expr.find_first_of(ops);
    if (op_pos == std::string::npos || op_pos == 0 || op_pos + 1 >= expr.size()) {
        return false;
    }

    const auto lhs = parse_integer(expr.substr(0, op_pos));
    const auto rhs = parse_integer(expr.substr(op_pos + 1));
    const auto answer = parse_integer(response);
    if (!lhs || !rhs || !answer) {
        return false;
    }

    long long expected = 0;
    switch (expr[op_pos]) {
    case '+': {
        if ((*rhs > 0 && *lhs > std::numeric_limits<long long>::max() - *rhs) ||
            (*rhs < 0 && *lhs < std::numeric_limits<long long>::min() - *rhs)) {
            return false;
        }
        expected = *lhs + *rhs;
        break;
    }
    case '-': {
        if ((*rhs > 0 && *lhs < std::numeric_limits<long long>::min() + *rhs) ||
            (*rhs < 0 && *lhs > std::numeric_limits<long long>::max() + *rhs)) {
            return false;
        }
        expected = *lhs - *rhs;
        break;
    }
    case '*': {
        if ((*lhs > 0 && *rhs > 0 &&
             *lhs > std::numeric_limits<long long>::max() / *rhs) ||
            (*lhs > 0 && *rhs < 0 &&
             *rhs < std::numeric_limits<long long>::min() / *lhs) ||
            (*lhs < 0 && *rhs > 0 &&
             *lhs < std::numeric_limits<long long>::min() / *rhs) ||
            (*lhs < 0 && *rhs < 0 &&
             *rhs < std::numeric_limits<long long>::max() / *lhs)) {
            return false;
        }
        expected = *lhs * *rhs;
        break;
    }
    case '/':
        if (*rhs == 0 ||
            (*lhs == std::numeric_limits<long long>::min() && *rhs == -1) ||
            (*lhs % *rhs) != 0) {
            return false;
        }
        expected = *lhs / *rhs;
        break;
    default:
        return false;
    }

    is_valid = (*answer == expected);
    return true;
}

bool InferenceEngine::self_heal(const std::string& prompt, const std::string& response) {
    bool is_valid = true;
    if (!try_evaluate_simple_math(prompt, response, is_valid)) {
        return false;
    }
    if (is_valid) {
        return false;
    }
    self_heal();
    return true;
}

void InferenceEngine::self_heal() {
    if (this->model) {
        this->model->reset_session();
    }
}

size_t InferenceEngine::get_memory_usage() const {
    if (!this->model) {
        return 0;
    }

    size_t bytes = 0;
    const auto add = [&](size_t amount) {
        if (amount > (std::numeric_limits<size_t>::max)() - bytes) {
            bytes = (std::numeric_limits<size_t>::max)();
        } else {
            bytes += amount;
        }
    };
    const auto add_tensor = [&](const Tensor& tensor) {
        if (tensor.size <= 0) return;
        const size_t elements = static_cast<size_t>(tensor.size);
        if (elements > (std::numeric_limits<size_t>::max)() / sizeof(float)) {
            add((std::numeric_limits<size_t>::max)());
        } else {
            add(elements * sizeof(float));
        }
    };

    std::unordered_set<const Parameter*> seen_parameters;
    for (auto* param : this->model->parameters()) {
        if (param && seen_parameters.insert(param).second) {
            add_tensor(param->data);
            add_tensor(param->grad);
        }
    }
    for (const BitLinear* layer : this->model->collect_bitlinear_layers()) {
        if (layer) add(layer->auxiliary_memory_usage_bytes());
    }
    if (this->trainer) {
        for (const auto& [unused, state] : this->trainer->m_state) add_tensor(state);
        for (const auto& [unused, state] : this->trainer->v_state) add_tensor(state);
        for (const auto& [unused, state] : this->trainer->quant_state) add(state.bytes());
    }
    return bytes;
}

} // namespace nsos
