#include "../include/nsos_sdk.h"
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
#include <mutex>
#include <optional>
#include <random>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../include/cuda/kernels.cuh"
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

bool parse_bool(const std::string& value, bool fallback = false) {
    const std::string lowered = trim_copy(value);
    if (lowered == "1" || lowered == "true" || lowered == "TRUE") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "FALSE") {
        return false;
    }
    return fallback;
}

constexpr uintmax_t kMaxConfigFileBytes = 1024 * 1024;
constexpr uintmax_t kMaxTokenizerPackBytes = 256ull * 1024ull * 1024ull;
constexpr uintmax_t kMaxEdgePackBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr uintmax_t kMaxWeightsPackBytes = 16ull * 1024ull * 1024ull * 1024ull;

std::filesystem::path atomic_temp_path(const std::filesystem::path& path) {
    return path.parent_path() /
           (path.filename().string() + ".tmp." +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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

std::string fnv1a_checksum_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Could not checksum file: " + path.string());
    }

    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;

    std::array<char, 4096> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
            hash *= kPrime;
        }
    }

    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

class Sha256Accumulator {
public:
    void update(const unsigned char* data, size_t size) {
        total_size_ += static_cast<uint64_t>(size);
        size_t cursor = 0;
        while (cursor < size) {
            const size_t to_copy = (std::min)(size - cursor, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, data + cursor, to_copy);
            buffered_ += to_copy;
            cursor += to_copy;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::string final_hex() {
        const uint64_t total_bits = total_size_ * 8ull;
        block_[buffered_++] = 0x80u;
        if (buffered_ > 56) {
            while (buffered_ < block_.size()) {
                block_[buffered_++] = 0;
            }
            transform(block_.data());
            buffered_ = 0;
        }
        while (buffered_ < 56) {
            block_[buffered_++] = 0;
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            block_[buffered_++] = static_cast<unsigned char>((total_bits >> shift) & 0xFFu);
        }
        transform(block_.data());
        buffered_ = 0;

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t value : state_) {
            out << std::setw(8) << value;
        }
        return out.str();
    }

private:
    static constexpr std::array<uint32_t, 64> kConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static uint32_t rotr(uint32_t value, uint32_t count) {
        return (value >> count) | (value << (32 - count));
    }

    static uint32_t choose(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    static uint32_t majority(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static uint32_t big_sigma0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }

    static uint32_t big_sigma1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    static uint32_t small_sigma0(uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }

    static uint32_t small_sigma1(uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    void transform(const unsigned char* block) {
        uint32_t w[64] = {};
        for (size_t i = 0; i < 16; ++i) {
            const size_t offset = i * 4;
            w[i] = (static_cast<uint32_t>(block[offset]) << 24) |
                   (static_cast<uint32_t>(block[offset + 1]) << 16) |
                   (static_cast<uint32_t>(block[offset + 2]) << 8) |
                   static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            w[i] = small_sigma1(w[i - 2]) + w[i - 7] + small_sigma0(w[i - 15]) + w[i - 16];
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t t1 = h + big_sigma1(e) + choose(e, f, g) + kConstants[i] + w[i];
            const uint32_t t2 = big_sigma0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<unsigned char, 64> block_ = {};
    size_t buffered_ = 0;
    uint64_t total_size_ = 0;
};

std::string sha256_checksum_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Could not hash file: " + path.string());
    }

    Sha256Accumulator accumulator;
    std::array<char, 4096> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            accumulator.update(reinterpret_cast<const unsigned char*>(buffer.data()),
                               static_cast<size_t>(count));
        }
    }
    return accumulator.final_hex();
}

void verify_pack_file_digest(const std::unordered_map<std::string, std::string>& manifest,
                             const std::filesystem::path& path,
                             const std::string& label,
                             const std::string& sha_key,
                             const std::string& fnv_key) {
    const auto sha_it = manifest.find(sha_key);
    if (sha_it != manifest.end()) {
        if (sha256_checksum_file(path) != sha_it->second) {
            throw std::runtime_error("Model pack " + label + " sha256 mismatch");
        }
        return;
    }

    const auto fnv_it = manifest.find(fnv_key);
    if (fnv_it != manifest.end() && fnv1a_checksum_file(path) != fnv_it->second) {
        throw std::runtime_error("Model pack " + label + " checksum mismatch");
    }
}

void write_model_config(const std::filesystem::path& path, const ModelConfig& config) {
    write_key_value_file(
        path,
        {
            {"num_layers", std::to_string(config.num_layers)},
            {"d_model", std::to_string(config.d_model)},
            {"vocab_size", std::to_string(config.vocab_size)},
            {"n_heads", std::to_string(config.n_heads)},
            {"n_kv_heads", std::to_string(config.n_kv_heads)},
            {"sliding_window", std::to_string(config.sliding_window)},
            {"attention_period", std::to_string(config.attention_period)},
            {"attention_slot", std::to_string(config.attention_slot)},
            {"num_experts", std::to_string(config.num_experts)},
            {"num_experts_per_token", std::to_string(config.num_experts_per_token)},
            {"use_moe", config.use_moe ? "true" : "false"},
            {"moe_period", std::to_string(config.moe_period)},
            {"moe_slot", std::to_string(config.moe_slot)},
            {"use_ttt", config.use_ttt ? "true" : "false"},
            {"ttt_period", std::to_string(config.ttt_period)},
            {"ttt_slot", std::to_string(config.ttt_slot)},
            {"use_gradient_checkpointing", config.use_gradient_checkpointing ? "true" : "false"},
            {"dropout", std::to_string(config.dropout)},
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
            {"chrass_density", std::to_string(config.chrass_density)},
            {"chrass_seed", std::to_string(config.chrass_seed)},
            {"logit_l2_beta", std::to_string(config.logit_l2_beta)},
            {"pantheon_vib_beta", std::to_string(config.pantheon_vib_beta)},
            {"use_slender_embedding", config.use_slender_embedding ? "true" : "false"},
            {"mamba_proper_ssm", config.mamba_proper_ssm ? "true" : "false"},
            {"mamba_state_expansion", config.mamba_state_expansion ? "true" : "false"},
            {"mamba_conv_kernel", std::to_string(config.mamba_conv_kernel)},
            {"mamba2_faithful", config.mamba2_faithful ? "true" : "false"},
            {"mamba_expand", std::to_string(config.mamba_expand)},
            {"mamba_head_dim", std::to_string(config.mamba_head_dim)},
            {"mamba_n_groups", std::to_string(config.mamba_n_groups)},
            {"tie_word_embeddings", config.tie_word_embeddings ? "true" : "false"},
            {"rope_theta", std::to_string(config.rope_theta)},
        });
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

    config.num_layers = get_int("num_layers", config.num_layers);
    config.d_model = get_int("d_model", config.d_model);
    config.vocab_size = get_int("vocab_size", config.vocab_size);
    config.n_heads = get_int("n_heads", config.n_heads);
    config.n_kv_heads = get_int("n_kv_heads", config.n_kv_heads);
    config.sliding_window = get_int("sliding_window", config.sliding_window);
    config.attention_period = get_int("attention_period", config.attention_period);
    config.attention_slot = get_int("attention_slot", config.attention_slot);
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
    config.tie_word_embeddings =
        get_bool("tie_word_embeddings", config.tie_word_embeddings);
    config.rope_theta = get_float("rope_theta", config.rope_theta);
    return config;
}

void validate_model_config_for_pack(const ModelConfig& config) {
    validate_model_config(config);
}

void apply_model_config_overrides(ModelConfig& base, const ModelConfig& overrides) {
    const ModelConfig defaults;
    if (overrides.num_layers != defaults.num_layers) base.num_layers = overrides.num_layers;
    if (overrides.d_model != defaults.d_model) base.d_model = overrides.d_model;
    if (overrides.vocab_size != defaults.vocab_size) base.vocab_size = overrides.vocab_size;
    if (overrides.n_heads != defaults.n_heads) base.n_heads = overrides.n_heads;
    if (overrides.n_kv_heads != defaults.n_kv_heads) base.n_kv_heads = overrides.n_kv_heads;
    if (overrides.sliding_window != defaults.sliding_window) base.sliding_window = overrides.sliding_window;
    if (overrides.attention_period != defaults.attention_period) {
        base.attention_period = overrides.attention_period;
    }
    if (overrides.attention_slot != defaults.attention_slot) {
        base.attention_slot = overrides.attention_slot;
    }
    if (overrides.num_experts != defaults.num_experts) base.num_experts = overrides.num_experts;
    if (overrides.num_experts_per_token != defaults.num_experts_per_token) {
        base.num_experts_per_token = overrides.num_experts_per_token;
    }
    if (overrides.use_moe != defaults.use_moe) base.use_moe = overrides.use_moe;
    if (overrides.moe_period != defaults.moe_period) base.moe_period = overrides.moe_period;
    if (overrides.moe_slot != defaults.moe_slot) base.moe_slot = overrides.moe_slot;
    if (overrides.use_ttt != defaults.use_ttt) base.use_ttt = overrides.use_ttt;
    if (overrides.ttt_period != defaults.ttt_period) base.ttt_period = overrides.ttt_period;
    if (overrides.ttt_slot != defaults.ttt_slot) base.ttt_slot = overrides.ttt_slot;
    if (overrides.use_gradient_checkpointing != defaults.use_gradient_checkpointing) {
        base.use_gradient_checkpointing = overrides.use_gradient_checkpointing;
    }
    if (std::abs(overrides.dropout - defaults.dropout) > 1e-6f) base.dropout = overrides.dropout;
    if (overrides.mcts_simulations != defaults.mcts_simulations) {
        base.mcts_simulations = overrides.mcts_simulations;
    }
    if (overrides.mcts_depth != defaults.mcts_depth) base.mcts_depth = overrides.mcts_depth;
    if (!overrides.checkpoint_path.empty() && overrides.checkpoint_path != defaults.checkpoint_path) {
        base.checkpoint_path = overrides.checkpoint_path;
    }
    if (overrides.max_context_tokens != defaults.max_context_tokens) {
        base.max_context_tokens = overrides.max_context_tokens;
    }
    if (overrides.default_batch_size != defaults.default_batch_size) {
        base.default_batch_size = overrides.default_batch_size;
    }
    if (overrides.use_cuda != defaults.use_cuda) base.use_cuda = overrides.use_cuda;
    if (overrides.use_exact_attention_training != defaults.use_exact_attention_training) {
        base.use_exact_attention_training = overrides.use_exact_attention_training;
    }
    if (overrides.use_flash_attn != defaults.use_flash_attn) {
        base.use_flash_attn = overrides.use_flash_attn;
    }
    if (overrides.moe_expert_hidden_dim != defaults.moe_expert_hidden_dim) {
        base.moe_expert_hidden_dim = overrides.moe_expert_hidden_dim;
    }
    if (overrides.use_kan != defaults.use_kan) base.use_kan = overrides.use_kan;
    if (overrides.use_chrass != defaults.use_chrass) base.use_chrass = overrides.use_chrass;
    if (std::abs(overrides.chrass_density - defaults.chrass_density) > 1e-6f) {
        base.chrass_density = overrides.chrass_density;
    }
    if (overrides.chrass_seed != defaults.chrass_seed) base.chrass_seed = overrides.chrass_seed;
    if (std::abs(overrides.logit_l2_beta - defaults.logit_l2_beta) > 1e-9f) {
        base.logit_l2_beta = overrides.logit_l2_beta;
    }
    if (std::abs(overrides.pantheon_vib_beta - defaults.pantheon_vib_beta) > 1e-9f) {
        base.pantheon_vib_beta = overrides.pantheon_vib_beta;
    }
    if (overrides.use_slender_embedding != defaults.use_slender_embedding) {
        base.use_slender_embedding = overrides.use_slender_embedding;
    }
    if (overrides.mamba_proper_ssm != defaults.mamba_proper_ssm) {
        base.mamba_proper_ssm = overrides.mamba_proper_ssm;
    }
    if (overrides.mamba_state_expansion != defaults.mamba_state_expansion) {
        base.mamba_state_expansion = overrides.mamba_state_expansion;
    }
    if (overrides.mamba_conv_kernel != defaults.mamba_conv_kernel) {
        base.mamba_conv_kernel = overrides.mamba_conv_kernel;
    }
    if (overrides.mamba2_faithful != defaults.mamba2_faithful) {
        base.mamba2_faithful = overrides.mamba2_faithful;
    }
    if (overrides.mamba_expand != defaults.mamba_expand) {
        base.mamba_expand = overrides.mamba_expand;
    }
    if (overrides.mamba_head_dim != defaults.mamba_head_dim) {
        base.mamba_head_dim = overrides.mamba_head_dim;
    }
    if (overrides.mamba_n_groups != defaults.mamba_n_groups) {
        base.mamba_n_groups = overrides.mamba_n_groups;
    }
    if (overrides.tie_word_embeddings != defaults.tie_word_embeddings) {
        base.tie_word_embeddings = overrides.tie_word_embeddings;
    }
    if (std::abs(overrides.rope_theta - defaults.rope_theta) > 1e-3f) {
        base.rope_theta = overrides.rope_theta;
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
    ~StreamingInferenceGuard() {
        if (model != nullptr) {
            try {
                model->set_streaming_inference(false);
            } catch (...) {
                // State cleanup must never replace an in-flight exception.
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
        // whichever staging path (pinned vs pageable) this generation used —
        // the empirical counterpart of nsos_bench_d2h_copy().
        if (d2h_copies_ > 0) {
            std::fprintf(stderr,
                         "[nsos] decode token D2H (%s staging): n=%zu "
                         "avg=%.2f us total=%.2f ms\n",
                         h_result_ ? "pinned" : "pageable", d2h_copies_,
                         d2h_us_total_ / static_cast<double>(d2h_copies_),
                         d2h_us_total_ / 1000.0);
        }
        free_all();
    }

    bool select(const float* raw_row_device, int vocab,
                const GenerationOptions& options,
                const std::vector<int>& output_tokens, size_t prompt_tokens_used,
                const std::function<bool(int, int)>& is_control_token,
                int& out_token) {
        if (vocab <= 0 || raw_row_device == nullptr) return false;
        if (!ensure(vocab)) return false;
        if (!control_built_ && !build_control(vocab, is_control_token)) return false;

        const size_t generated_so_far =
            output_tokens.size() > prompt_tokens_used
                ? (output_tokens.size() - prompt_tokens_used)
                : 0;

        // repeated mask: incrementally mark the generated tokens (output[prompt:])
        // so the penalty matches the host's per-token "repeated" set.
        for (size_t i = prompt_tokens_used + marked_count_;
             i < output_tokens.size(); ++i) {
            const int t = output_tokens[i];
            if (t >= 0 && t < vocab) {
                if (cudaMemsetAsync(d_repeated_ + t, 1, 1, 0) != cudaSuccess) {
                    return false;
                }
            }
        }
        marked_count_ = generated_so_far;

        // seen mask: zero, then scatter no-repeat-ngram bans + blocked-EOS.
        if (cudaMemsetAsync(d_seen_, 0, static_cast<size_t>(vocab), 0) != cudaSuccess) {
            return false;
        }
        if (generated_so_far > 0 && options.no_repeat_ngram_size > 1) {
            compute_no_repeat_ngram_banned(output_tokens, prompt_tokens_used,
                                           options.no_repeat_ngram_size,
                                           generated_so_far, banned_);
            for (int t : banned_) {
                if (t >= 0 && t < vocab) {
                    if (cudaMemsetAsync(d_seen_ + t, 1, 1, 0) != cudaSuccess) {
                        return false;
                    }
                }
            }
        }
        const bool block_eos =
            generated_so_far <
                static_cast<size_t>(std::max(options.min_new_tokens, 0)) &&
            options.eos_token_id >= 0 && options.eos_token_id < vocab;
        if (block_eos) {
            if (cudaMemsetAsync(d_seen_ + options.eos_token_id, 1, 1, 0) !=
                cudaSuccess) {
                return false;
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
                                    penalty_active ? d_repeated_ : nullptr,
                                    d_seen_, d_control_,
                                    suppress_control ? 1 : 0, penalty, d_result_);
        if (cudaGetLastError() != cudaSuccess) return false;
        // D2H de 4 bytes por token no staging PINNED (h_result_): memoria
        // pageable forca o driver a passar por um buffer intermediario; pinned
        // faz DMA direto — no caminho por-token do decode isso remove uma copia
        // extra por passo (item do estudo CuPy).
        int token = options.eos_token_id;
        int* dst = h_result_ ? h_result_ : &token;
        static const bool d2h_timing = [] {
            const char* v = std::getenv("NSOS_D2H_TIMING");
            return v != nullptr && v[0] == '1';
        }();
        std::chrono::steady_clock::time_point d2h_started;
        if (d2h_timing) d2h_started = std::chrono::steady_clock::now();
        if (cudaMemcpy(dst, d_result_, sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        if (d2h_timing) {
            d2h_us_total_ += static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - d2h_started)
                    .count()) /
                1000.0;
            ++d2h_copies_;
        }
        if (h_result_) token = *h_result_;
        // -1 sentinel: every candidate was masked.  Fall back to the host greedy
        // branch, which scans for the first allowed token then EOS/0 (the GPU
        // kernel cannot reproduce that scan), so behaviour matches exactly.
        if (token < 0 || token >= vocab) return false;
        out_token = token;
        return true;
    }

private:
    bool ensure(int vocab) {
        if (vocab_ == vocab && d_seen_ != nullptr) return true;
        free_all();
        const size_t bytes = static_cast<size_t>(vocab);
        if (cudaMalloc(&d_repeated_, bytes) != cudaSuccess ||
            cudaMalloc(&d_seen_, bytes) != cudaSuccess ||
            cudaMalloc(&d_control_, bytes) != cudaSuccess ||
            cudaMalloc(&d_result_, sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            free_all();
            return false;
        }
        // Staging host PINNED para o D2H por-token (opcional: falha degrada
        // para o caminho pageable via &token, nunca aborta o sampler).
        if (cudaMallocHost(&h_result_, sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            h_result_ = nullptr;
        }
        if (cudaMemset(d_repeated_, 0, bytes) != cudaSuccess ||
            cudaMemset(d_control_, 0, bytes) != cudaSuccess) {
            free_all();
            return false;
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
        if (cudaMemcpy(d_control_, host.data(), static_cast<size_t>(vocab),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        control_built_ = true;
        return true;
    }
    void free_all() {
        if (d_repeated_) cudaFree(d_repeated_);
        if (d_seen_) cudaFree(d_seen_);
        if (d_control_) cudaFree(d_control_);
        if (d_result_) cudaFree(d_result_);
        if (h_result_) cudaFreeHost(h_result_);
        d_repeated_ = d_seen_ = d_control_ = nullptr;
        d_result_ = nullptr;
        h_result_ = nullptr;
        vocab_ = 0;
        control_built_ = false;
        marked_count_ = 0;
    }
    unsigned char* d_repeated_ = nullptr;
    unsigned char* d_seen_ = nullptr;
    unsigned char* d_control_ = nullptr;
    int* d_result_ = nullptr;
    int* h_result_ = nullptr;  // pinned staging p/ o D2H por-token
    int vocab_ = 0;
    bool control_built_ = false;
    size_t marked_count_ = 0;
    std::vector<int> banned_;
    double d2h_us_total_ = 0.0;  // NSOS_D2H_TIMING accumulators
    size_t d2h_copies_ = 0;
};
#endif  // USE_CUDA

} // namespace

bool InferenceEngine::try_load_model_pack(const std::string& path,
                                          const ModelConfig& runtime_overrides) {
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
    const fs::path edge_linear_path =
        has_edge_linear ? pack_child_path(pack_root, get_value("edge_linear"), "edge_linear")
                        : fs::path{};

    ensure_regular_file_within_limit(config_path, kMaxConfigFileBytes, "config");
    ensure_regular_file_within_limit(tokenizer_path, kMaxTokenizerPackBytes, "tokenizer");
    ensure_regular_file_within_limit(weights_path, kMaxWeightsPackBytes, "weights");
    if (has_edge_linear) {
        ensure_regular_file_within_limit(edge_linear_path, kMaxEdgePackBytes, "edge linear");
    }

    verify_pack_file_digest(manifest, config_path, "config", "sha256_config", "checksum_config");
    verify_pack_file_digest(manifest, tokenizer_path, "tokenizer", "sha256_tokenizer",
                            "checksum_tokenizer");
    verify_pack_file_digest(manifest, weights_path, "weights", "sha256_weights",
                            "checksum_weights");
    if (has_edge_linear) {
        verify_pack_file_digest(manifest, edge_linear_path, "edge linear", "sha256_edge_linear",
                                "checksum_edge_linear");
    }

    ModelConfig pack_config = read_model_config(config_path);
    apply_model_config_overrides(pack_config, runtime_overrides);
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
        // VISION #3 (Quantized Inference): when an edge_linear pack is
        // available in the bundle, the WHOLE POINT of the project is to
        // serve from 1.58-bit packed weights at inference, not from
        // residual FP32 weights.  The historical default here was
        // `false` (keep FP32 alongside the edge pack, use FP32 in
        // forward, treat edge pack as a reference for the audit hook),
        // which meant the 1.58-bit edge claim was a memory format
        // statement only — actual serving still cost FP32 RAM + bandwidth.
        //
        // The right default is to RELEASE FP32 after loading the edge
        // pack, so the runtime is genuinely operating in ternary.  Users
        // who need the FP32 reference path for parity testing can opt
        // back in by setting NSOS_KEEP_FP32_WEIGHTS=1 in the environment.
        bool keep_fp32 = false;
        if (const char* env = std::getenv("NSOS_KEEP_FP32_WEIGHTS")) {
            std::string s(env);
            keep_fp32 = (s == "1" || s == "true" || s == "TRUE" || s == "yes");
        }
        const bool release_fp32 = !keep_fp32;
        this->model->load_edge_linear_pack(edge_linear_path.string(), release_fp32);
        if (release_fp32) {
            std::cerr << "[InferenceEngine] edge pack loaded; FP32 linear weights "
                      << "released (1.58-bit inference mode).  Set "
                      << "NSOS_KEEP_FP32_WEIGHTS=1 to keep both."
                      << std::endl;
        } else {
            std::cerr << "[InferenceEngine] edge pack loaded; FP32 weights retained "
                      << "(reference mode, opt-in via NSOS_KEEP_FP32_WEIGHTS=1)."
                      << std::endl;
        }
    }
    this->loaded_from_pack_ = true;
    return true;
}

bool InferenceEngine::load_model(const std::string& path, const ModelConfig& config_value) {
    auto commit = [&](InferenceEngine&& staged) {
        this->model = std::move(staged.model);
        this->trainer = std::move(staged.trainer);
        this->tokenizer = std::move(staged.tokenizer);
        this->config = staged.config;
        this->last_metrics_ = {};
        this->loaded_from_pack_ = staged.loaded_from_pack_;
    };

    try {
        if (!path.empty()) {
            InferenceEngine staged_pack;
            if (staged_pack.try_load_model_pack(path, config_value)) {
                commit(std::move(staged_pack));
                return true;
            }
        }

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
        staged.config = config_value;
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
        commit(std::move(staged));
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[InferenceEngine] Failed to load model from '" << path
                  << "': " << ex.what() << std::endl;
        return false;
    }
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

    const int context_limit = std::max(
        options.max_context_tokens > 0 ? options.max_context_tokens : this->config.max_context_tokens,
        1);
    std::vector<int> output = prompt_tokens;
    if (static_cast<int>(output.size()) > context_limit) {
        output.erase(output.begin(), output.end() - context_limit);
    }
    const size_t prompt_tokens_used = output.size();
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
            }
        }
#endif
        if (logits.shape.empty()) {
            throw std::runtime_error("Model returned logits without a vocabulary dimension");
        }
        Tensor host_logits = (logits.get_device() == Device::GPU) ? logits.cpu() : logits;
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
    StreamingInferenceGuard streaming_guard{this->model.get()};

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
            // Device: take it from the model's first parameter that
            // exposes a device.  Inference engines set this consistently
            // at load_model time.  If we can't determine the device,
            // skip the reserve — the cache will grow on demand as before.
            try {
                const Device device = this->model->parameters().empty()
                    ? Device::CPU
                    : this->model->parameters().front()->data.get_device();
                this->model->reserve_kv_cache(reserve_total, device, 1);
            } catch (const std::exception&) {
                // Reserve is an optimization, never a correctness
                // requirement.  Failures fall through to on-demand
                // growth.
            }
        }
    }

    auto prefill_started_at = std::chrono::steady_clock::now();
    auto decode_started_at = prefill_started_at;
    auto decode_finished_at = prefill_started_at;
    if (can_use_streaming) {
        Tensor logits;
        if (!output.empty()) {
            logits = this->model->forward_ids(output, nullptr);
        }
        decode_started_at = std::chrono::steady_clock::now();

        // CUDA-graph decode (opt-in NSOS_CUDA_GRAPH_DECODE=1 on an
        // NSOS_CUDA_PTDS build): the model captures one single-token forward
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

            if (try_decode_graph) {
                Tensor graphed = this->model->forward_ids_decode_graph(next_token);
                if (graphed.size > 0) {
                    logits = std::move(graphed);
                    continue;
                }
                try_decode_graph = false;
            }
            logits = this->model->forward_ids({next_token}, nullptr);
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
            Tensor logits = this->model->forward_ids(model_input, nullptr);
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
    last_metrics_.mamba_last_fallback_reason = runtime.mamba_last_fallback_reason;
    return result;
}

std::vector<std::string> InferenceEngine::generate_batch(
    const std::vector<std::string>& prompts,
    const GenerationOptions& options) {
    if (!this->model) {
        throw std::runtime_error("No model loaded");
    }
    validate_generation_options(options, this->config.vocab_size);

    auto started_at = std::chrono::steady_clock::now();
    if (prompts.empty()) {
        last_metrics_ = {};
        return {};
    }
    constexpr size_t kMaxDirectBatchPrompts = 1024;
    constexpr size_t kMaxDirectBatchBytes = 64ull * 1024ull * 1024ull;
    if (prompts.size() > kMaxDirectBatchPrompts) {
        throw std::invalid_argument("prompt batch exceeds the direct SDK item limit");
    }
    size_t total_prompt_bytes = 0;
    for (const auto& prompt : prompts) {
        if (prompt.size() > kMaxDirectBatchBytes - total_prompt_bytes) {
            throw std::invalid_argument("prompt batch exceeds the direct SDK byte limit");
        }
        total_prompt_bytes += prompt.size();
    }

    struct BatchItem {
        std::vector<int> output_tokens;
        size_t prompt_tokens_used = 0;
        bool finished = false;
        Tensor cached_logits;
        JambaSessionSnapshot snapshot;
        bool has_snapshot = false;
        std::mt19937 rng;
        SamplerWorkspace sampler_workspace;
    };

    GenerationMetrics aggregate{};
    aggregate.batch_size = prompts.size();
    aggregate.loaded_from_pack = loaded_from_pack_;
    std::vector<std::string> outputs(prompts.size());
    std::vector<BatchItem> items(prompts.size());

    const int context_limit = std::max(
        options.max_context_tokens > 0 ? options.max_context_tokens : this->config.max_context_tokens,
        1);
    const int top_k =
        options.top_k > 0
            ? options.top_k
            : std::min(std::max(this->config.vocab_size / 8, 8),
                       std::max(this->config.vocab_size - 1, 1));
    this->model->set_training_mode(false);
    this->model->reset_runtime_telemetry();
    const bool can_use_streaming = this->model->supports_streaming_inference();
    aggregate.used_streaming = can_use_streaming;
    this->model->set_streaming_inference(can_use_streaming);
    StreamingInferenceGuard streaming_guard{this->model.get()};
    this->model->reset_session();

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

    auto sample_next_token =
        [&](const float* raw, int vocab_size, BatchItem& item) {
            return sample_from_host_logits_row(
                raw,
                vocab_size,
                top_k,
                options,
                item.output_tokens,
                item.prompt_tokens_used,
                item.rng,
                item.sampler_workspace,
                [&](int token, int current_vocab_size) {
                    return cached_token_piece(token, current_vocab_size).rfind("<|", 0) == 0;
                },
                &aggregate.sampler_ms);
        };

    for (size_t index = 0; index < prompts.size(); ++index) {
        items[index].output_tokens = sanitize_token_ids(this->tokenizer.encode(prompts[index]));
        if (items[index].output_tokens.empty()) {
            items[index].output_tokens.push_back(1);
        }
        const size_t original_prompt_tokens = items[index].output_tokens.size();
        aggregate.prompt_tokens_total += original_prompt_tokens;
        if (static_cast<int>(items[index].output_tokens.size()) > context_limit) {
            items[index].output_tokens.erase(
                items[index].output_tokens.begin(),
                items[index].output_tokens.end() - context_limit);
        }
        items[index].prompt_tokens_used = items[index].output_tokens.size();
        const uint64_t item_sequence =
            fnv1a_hash_text(prompts[index]) ^
            (static_cast<uint64_t>(std::max(options.max_tokens, 0)) << 32) ^
            static_cast<uint64_t>(std::max(options.eos_token_id, 0));
        items[index].rng = make_sampler_rng("generate_batch_item", item_sequence);
    }

    for (const auto& item : items) {
        aggregate.prompt_tokens_used += item.prompt_tokens_used;
    }
    this->model->record_audit_token_context(items.front().output_tokens,
                                            items.size(),
                                            aggregate.prompt_tokens_total,
                                            aggregate.prompt_tokens_used,
                                            context_limit,
                                            aggregate.prompt_tokens_used <
                                                aggregate.prompt_tokens_total);

    auto prefill_started_at = std::chrono::steady_clock::now();
    auto decode_started_at = prefill_started_at;
    auto decode_finished_at = prefill_started_at;
    const bool equal_stream_lengths =
        std::all_of(items.begin(), items.end(), [&](const auto& item) {
            return item.prompt_tokens_used == items.front().prompt_tokens_used;
        });
    if (can_use_streaming && this->model->supports_batched_streaming_inference() &&
        equal_stream_lengths) {
        auto to_host_logits = [](const Tensor& logits) {
            return logits.get_device() == Device::GPU ? logits.cpu() : logits;
        };

        for (size_t item_index = 0; item_index < items.size(); ++item_index) {
            this->model->reset_session();
            this->model->set_streaming_inference(true);
            Tensor logits = this->model->forward_ids(items[item_index].output_tokens, nullptr);
            Tensor host_logits = to_host_logits(logits);
            if (host_logits.shape.empty()) {
                throw std::runtime_error("Model returned logits without a vocabulary dimension");
            }
            const int vocab_size = host_logits.shape.back();
            if (vocab_size <= 0 || vocab_size > host_logits.size) {
                throw std::runtime_error("Model returned an invalid prefill logits shape");
            }
            const int last_offset = host_logits.size - vocab_size;
            Tensor row_logits({1, vocab_size}, Device::CPU);
            std::memcpy(row_logits.data(),
                        host_logits.data() + last_offset,
                        static_cast<size_t>(vocab_size) * sizeof(float));
            items[item_index].cached_logits = std::move(row_logits);
            items[item_index].snapshot = this->model->fork_session();
            items[item_index].has_snapshot = true;
        }
        decode_started_at = std::chrono::steady_clock::now();

        for (int step = 0; step < std::max(options.max_tokens, 0); ++step) {
            std::vector<size_t> active_indices;
            std::vector<std::vector<int>> batch_next_tokens;
            active_indices.reserve(items.size());
            batch_next_tokens.reserve(items.size());
            for (size_t index = 0; index < items.size(); ++index) {
                if (items[index].finished) {
                    continue;
                }
                const Tensor& host_logits = items[index].cached_logits;
                const int vocab_size = host_logits.shape.back();
                const size_t last_offset = static_cast<size_t>(host_logits.size - vocab_size);
                const int next_token = sample_next_token(
                    host_logits.data() + last_offset, vocab_size, items[index]);
                items[index].output_tokens.push_back(next_token);
                ++aggregate.generated_tokens;
                if (next_token == options.eos_token_id) {
                    items[index].finished = true;
                    continue;
                }
                if (!items[index].has_snapshot) {
                    throw std::runtime_error("Streaming batch item lost session snapshot");
                }
                active_indices.push_back(index);
                batch_next_tokens.push_back({next_token});
            }
            if (active_indices.empty()) {
                break;
            }

            std::vector<JambaSessionSnapshot> active_snapshots;
            active_snapshots.reserve(active_indices.size());
            for (size_t index : active_indices) {
                active_snapshots.push_back(items[index].snapshot);
            }

            this->model->restore_session_batch(active_snapshots);
            Tensor next_logits_batch = this->model->forward_ids_batch(batch_next_tokens, nullptr);
            Tensor host_logits_batch = to_host_logits(next_logits_batch);
            if (host_logits_batch.shape.size() != 3) {
                throw std::runtime_error("Model returned invalid batched streaming logits rank");
            }
            const int batch_count = host_logits_batch.shape[0];
            const int max_seq_len = host_logits_batch.shape[1];
            const int vocab_size = host_logits_batch.shape[2];
            const float* logits_ptr = host_logits_batch.data();
            auto updated_snapshots = this->model->fork_session_batch();
            if (batch_count != static_cast<int>(active_indices.size()) || max_seq_len <= 0 ||
                vocab_size <= 0 || updated_snapshots.size() != active_indices.size()) {
                throw std::runtime_error("Model returned inconsistent batched streaming state");
            }

            for (int row = 0; row < batch_count; ++row) {
                const size_t item_index = active_indices[static_cast<size_t>(row)];
                const size_t row_offset =
                    ((static_cast<size_t>(row) * static_cast<size_t>(max_seq_len)) +
                     static_cast<size_t>(max_seq_len - 1)) *
                    static_cast<size_t>(vocab_size);
                Tensor row_logits({1, vocab_size}, Device::CPU);
                std::memcpy(row_logits.data(),
                            logits_ptr + row_offset,
                            static_cast<size_t>(vocab_size) * sizeof(float));
                items[item_index].cached_logits = std::move(row_logits);
                items[item_index].snapshot = updated_snapshots[static_cast<size_t>(row)];
                items[item_index].has_snapshot = true;
            }
        }
        decode_finished_at = std::chrono::steady_clock::now();
    } else if (can_use_streaming) {
        auto to_host_logits = [](const Tensor& logits) {
            return logits.get_device() == Device::GPU ? logits.cpu() : logits;
        };

        for (size_t index = 0; index < items.size(); ++index) {
            this->model->reset_session();
            this->model->set_streaming_inference(true);
            Tensor logits = this->model->forward_ids(items[index].output_tokens, nullptr);
            items[index].cached_logits = to_host_logits(logits);
            if (items[index].cached_logits.shape.empty() ||
                items[index].cached_logits.shape.back() <= 0 ||
                items[index].cached_logits.shape.back() > items[index].cached_logits.size) {
                throw std::runtime_error("Model returned an invalid prefill logits shape");
            }
            items[index].snapshot = this->model->fork_session();
            items[index].has_snapshot = true;
        }
        decode_started_at = std::chrono::steady_clock::now();

        for (int step = 0; step < std::max(options.max_tokens, 0); ++step) {
            bool any_active = false;
            for (size_t index = 0; index < items.size(); ++index) {
                if (items[index].finished) {
                    continue;
                }
                any_active = true;
                const Tensor& host_logits = items[index].cached_logits;
                const int vocab_size = host_logits.shape.back();
                const size_t last_offset = static_cast<size_t>(host_logits.size - vocab_size);
                const int next_token = sample_next_token(
                    host_logits.data() + last_offset, vocab_size, items[index]);
                items[index].output_tokens.push_back(next_token);
                ++aggregate.generated_tokens;
                if (next_token == options.eos_token_id) {
                    items[index].finished = true;
                    continue;
                }
                if (!items[index].has_snapshot) {
                    throw std::runtime_error("Streaming batch item lost session snapshot");
                }
                this->model->restore_session(items[index].snapshot);
                Tensor next_logits = this->model->forward_ids({next_token}, nullptr);
                items[index].cached_logits = to_host_logits(next_logits);
                items[index].snapshot = this->model->fork_session();
            }
            if (!any_active) {
                break;
            }
        }
        decode_finished_at = std::chrono::steady_clock::now();
    } else {
        decode_started_at = std::chrono::steady_clock::now();
        for (int step = 0; step < std::max(options.max_tokens, 0); ++step) {
            std::vector<size_t> active_indices;
            std::vector<std::vector<int>> batch_inputs;
            active_indices.reserve(items.size());
            batch_inputs.reserve(items.size());

            for (size_t index = 0; index < items.size(); ++index) {
                if (items[index].finished) {
                    continue;
                }
                active_indices.push_back(index);
                std::vector<int> model_input = items[index].output_tokens;
                if (static_cast<int>(model_input.size()) > context_limit) {
                    model_input.erase(model_input.begin(), model_input.end() - context_limit);
                }
                batch_inputs.push_back(std::move(model_input));
            }

            if (active_indices.empty()) {
                break;
            }

            this->model->reset_session();
            Tensor logits_batch = this->model->forward_ids_batch(batch_inputs, nullptr);
            Tensor host_logits = (logits_batch.get_device() == Device::GPU) ? logits_batch.cpu()
                                                                            : logits_batch;
            if (host_logits.shape.size() != 3) {
                throw std::runtime_error("Model returned invalid batched logits rank");
            }
            const int batch_count = host_logits.shape[0];
            const int max_seq_len = host_logits.shape[1];
            const int vocab_size = host_logits.shape[2];
            const float* logits_ptr = host_logits.data();
            if (batch_count != static_cast<int>(active_indices.size()) || max_seq_len <= 0 ||
                vocab_size <= 0) {
                throw std::runtime_error("Model returned inconsistent batched logits shape");
            }

            for (int row = 0; row < batch_count; ++row) {
                const size_t item_index = active_indices[static_cast<size_t>(row)];
                const int valid_len = static_cast<int>(batch_inputs[static_cast<size_t>(row)].size());
                if (valid_len <= 0 || valid_len > max_seq_len) {
                    throw std::runtime_error("Model returned invalid batched sequence padding");
                }
                const size_t row_offset =
                    ((static_cast<size_t>(row) * max_seq_len) + static_cast<size_t>(valid_len - 1)) *
                    static_cast<size_t>(vocab_size);
                const int next_token = sample_next_token(
                    logits_ptr + row_offset, vocab_size, items[item_index]);
                items[item_index].output_tokens.push_back(next_token);
                ++aggregate.generated_tokens;
                if (next_token == options.eos_token_id) {
                    items[item_index].finished = true;
                }
            }
        }
        decode_finished_at = std::chrono::steady_clock::now();
    }

    this->model->set_streaming_inference(false);

    for (size_t index = 0; index < items.size(); ++index) {
        for (size_t token_index = items[index].prompt_tokens_used;
             token_index < items[index].output_tokens.size();
             ++token_index) {
            if (items[index].output_tokens[token_index] == options.eos_token_id) {
                continue;
            }
            outputs[index] += decode_token_piece(this->tokenizer,
                                                 items[index].output_tokens[token_index]);
        }
    }

    const auto finished_at = std::chrono::steady_clock::now();
    aggregate.elapsed_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(finished_at - started_at)
                .count());
    aggregate.elapsed_ms /= 1000.0;
    aggregate.prefill_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(decode_started_at -
                                                                  prefill_started_at)
                .count()) /
        1000.0;
    aggregate.decode_ms =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(decode_finished_at -
                                                                  decode_started_at)
                .count()) /
        1000.0;
    const double elapsed_seconds = std::max(aggregate.elapsed_ms / 1000.0, 1e-9);
    const double prefill_seconds = std::max(aggregate.prefill_ms / 1000.0, 1e-9);
    const double decode_seconds = std::max(aggregate.decode_ms / 1000.0, 1e-9);
    aggregate.prompt_tokens_per_sec =
        static_cast<double>(aggregate.prompt_tokens_used) /
        (can_use_streaming ? prefill_seconds : elapsed_seconds);
    aggregate.decode_tokens_per_sec =
        static_cast<double>(aggregate.generated_tokens) /
        (aggregate.decode_ms > 0.0 ? decode_seconds : elapsed_seconds);
    aggregate.total_tokens_per_sec =
        static_cast<double>(aggregate.prompt_tokens_used + aggregate.generated_tokens) /
        elapsed_seconds;
    const RuntimeTelemetrySnapshot runtime = this->model->runtime_telemetry();
    aggregate.mamba_fast_path_hits = runtime.mamba_fast_path_hits;
    aggregate.mamba_fast_path_fallbacks = runtime.mamba_fast_path_fallbacks;
    aggregate.mamba_last_fallback_reason = runtime.mamba_last_fallback_reason;
    last_metrics_ = aggregate;
    return outputs;
}

float InferenceEngine::train_step(const std::vector<int>& input,
                                  const std::vector<int>& target) {
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

bool InferenceEngine::save_checkpoint(const std::string& path) const {
    if (!this->model || path.empty()) {
        return false;
    }
    const std::filesystem::path destination(path);
    const std::filesystem::path temporary = atomic_temp_path(destination);
    try {
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
        fs::create_directories(pack_root);

        const fs::path weights_path = pack_root / "model.nsos.bin";
        const fs::path tokenizer_path = pack_root / "tokenizer.nsos";
        const fs::path config_path = pack_root / "config.nsos";
        const fs::path edge_linear_path = pack_root / "edge_linear.nsos";
        const fs::path manifest_path = pack_root / "manifest.nsos";

        const fs::path weights_temp = atomic_temp_path(weights_path);
        const fs::path edge_linear_temp = atomic_temp_path(edge_linear_path);
        const fs::path tokenizer_temp = atomic_temp_path(tokenizer_path);
        try {
            this->model->save(weights_temp.string());
            replace_file(weights_temp, weights_path);
            this->model->save_edge_linear_pack(edge_linear_temp.string());
            replace_file(edge_linear_temp, edge_linear_path);
            this->tokenizer.save_pack(tokenizer_temp.string());
            replace_file(tokenizer_temp, tokenizer_path);
            write_model_config(config_path, this->config);

            // Publish the manifest last. Readers either see the previous
            // manifest (and reject a mismatched partial generation) or the
            // complete new generation; they never accept unchecked children.
            write_key_value_file(
                manifest_path,
                {
                    {"format", "nsos-pack-v2"},
                    {"version", "2"},
                    {"weights", weights_path.filename().string()},
                    {"edge_linear", edge_linear_path.filename().string()},
                    {"tokenizer", tokenizer_path.filename().string()},
                    {"config", config_path.filename().string()},
                    {"checksum_weights", fnv1a_checksum_file(weights_path)},
                    {"checksum_edge_linear", fnv1a_checksum_file(edge_linear_path)},
                    {"checksum_tokenizer", fnv1a_checksum_file(tokenizer_path)},
                    {"checksum_config", fnv1a_checksum_file(config_path)},
                    {"sha256_weights", sha256_checksum_file(weights_path)},
                    {"sha256_edge_linear", sha256_checksum_file(edge_linear_path)},
                    {"sha256_tokenizer", sha256_checksum_file(tokenizer_path)},
                    {"sha256_config", sha256_checksum_file(config_path)},
                });
        } catch (...) {
            std::error_code ignored;
            fs::remove(weights_temp, ignored);
            fs::remove(edge_linear_temp, ignored);
            fs::remove(tokenizer_temp, ignored);
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

    auto replica = std::make_unique<InferenceEngine>();
    replica->config = this->config;
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
        dst->import_packed_state(state, device, !src->has_full_precision_weight());
        dst->set_reference_path(src->reference_path_enabled());
    }

    replica->last_metrics_ = this->last_metrics_;
    return replica;
}

std::unique_ptr<InferenceEngine> InferenceEngine::clone_for_training() const {
    if (!this->model || !this->trainer) {
        throw std::runtime_error("Cannot clone training engine without a loaded trainer");
    }

    auto clone = clone_for_inference();
    clone->model->set_training_mode(true);
    clone->model->set_streaming_inference(false);
    clone->trainer = std::make_unique<Trainer>(clone->model.get(), this->trainer->learning_rate);

    const Trainer& source = *this->trainer;
    Trainer& target = *clone->trainer;
    target.beta1 = source.beta1;
    target.beta2 = source.beta2;
    target.eps = source.eps;
    target.weight_decay = source.weight_decay;
    target.max_grad_norm = source.max_grad_norm;
    target.min_learning_rate_scale = source.min_learning_rate_scale;
    target.first_token_loss_scale = source.first_token_loss_scale;
    target.eos_loss_scale = source.eos_loss_scale;
    target.repetition_unlikelihood_scale = source.repetition_unlikelihood_scale;
    target.moe_aux_loss_scale = source.moe_aux_loss_scale;
    target.pantheon_vib_beta = source.pantheon_vib_beta;
    target.logit_l2_beta = source.logit_l2_beta;
    target.warmup_steps = source.warmup_steps;
    target.global_step_count = source.global_step_count;
    target.total_training_steps = source.total_training_steps;
    target.eos_token_id = source.eos_token_id;
    target.optimizer_state_bits = source.optimizer_state_bits;
    target.phase_scheduler = source.phase_scheduler;
    target.last_auxiliary_stats = source.last_auxiliary_stats;
    target.last_objective_stats = source.last_objective_stats;

    const auto source_parameters = this->model->parameters();
    const auto target_parameters = clone->model->parameters();
    if (source_parameters.size() != target_parameters.size()) {
        throw std::runtime_error("Training clone parameter count mismatch");
    }
    for (size_t index = 0; index < source_parameters.size(); ++index) {
        Parameter* src = source_parameters[index];
        Parameter* dst = target_parameters[index];
        if (!src || !dst) {
            throw std::runtime_error("Training clone encountered null parameter");
        }
        if (src->data.size == 0) {
            throw std::runtime_error(
                "Training cannot start from an inference-only packed model");
        }
        if (src->grad.size > 0) {
            if (dst->grad.size == 0 || dst->grad.shape != src->grad.shape) {
                dst->grad = src->grad.clone();
            } else {
                dst->grad.copy_from(src->grad);
            }
        } else {
            dst->grad = Tensor();
        }

        if (const auto it = source.m_state.find(src); it != source.m_state.end()) {
            target.m_state.emplace(dst, it->second.clone());
        }
        if (const auto it = source.v_state.find(src); it != source.v_state.end()) {
            target.v_state.emplace(dst, it->second.clone());
        }
        if (const auto it = source.quant_state.find(src); it != source.quant_state.end()) {
            target.quant_state.emplace(dst, it->second);
        }
        if (const auto it = source.crit_g0_state.find(src); it != source.crit_g0_state.end()) {
            target.crit_g0_state.emplace(dst, it->second);
        }
        if (const auto it = source.external_lr_scale.find(src);
            it != source.external_lr_scale.end()) {
            target.external_lr_scale.emplace(dst, it->second);
        }
        if (const auto it = source.criticality_lr_scale.find(src);
            it != source.criticality_lr_scale.end()) {
            target.criticality_lr_scale.emplace(dst, it->second);
        }
    }

    if (target.m_state.size() != source.m_state.size() ||
        target.v_state.size() != source.v_state.size() ||
        target.quant_state.size() != source.quant_state.size() ||
        target.crit_g0_state.size() != source.crit_g0_state.size() ||
        target.external_lr_scale.size() != source.external_lr_scale.size() ||
        target.criticality_lr_scale.size() != source.criticality_lr_scale.size()) {
        throw std::runtime_error("Training clone optimizer state referenced an unknown parameter");
    }
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
