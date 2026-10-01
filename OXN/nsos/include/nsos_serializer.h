#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include "nsos/sha256.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nsos {

#ifdef NSOS_ENABLE_TEST_HOOKS
namespace testing {
// Allows `countdown` target-device staging operations, then injects
// std::bad_alloc before the next one. A negative value disables injection.
void set_model_checkpoint_stage_failure_countdown(long long countdown);
void clear_model_checkpoint_stage_failure();
}  // namespace testing
namespace detail {
void model_checkpoint_stage_fault_point();
}  // namespace detail
#endif

class ModelSerializer {
public:
    // v2 fingerprint from the model's live config.  A-domain is always log in
    // v2 (the N1 reparameterization is unconditional in current code).
    static uint32_t architecture_fingerprint(const JambaModel* model) {
        const ModelConfig& cfg = model->model_config();
        uint32_t fp = NSOS_FP_A_LOG_DOMAIN;
        if (cfg.mamba_proper_ssm) fp |= NSOS_FP_MAMBA_PROPER;
        if (cfg.mamba_state_expansion) fp |= NSOS_FP_STATE_EXPANSION;
        if (cfg.tie_word_embeddings) fp |= NSOS_FP_TIE_EMBEDDINGS;
        if (cfg.mamba2_faithful) fp |= NSOS_FP_MAMBA2_FAITHFUL;
        if (cfg.mamba3_enabled) fp |= NSOS_FP_MAMBA3;
        if (cfg.hybrid_composition ==
            HybridComposition::ParallelGated) {
            fp |= NSOS_FP_HYBRID_PARALLEL;
        }
        if (cfg.faithful_attention_linears) {
            fp |= NSOS_FP_EXACT_ATTN_LINEAR;
        }
        if (cfg.force_mamba_last_layer) {
            fp |= NSOS_FP_FORCE_MAMBA_LAST;
        }
        return fp;
    }

    static std::string configuration_digest(const JambaModel* model) {
        std::vector<unsigned char> canonical;
        canonical.reserve(192);
        auto append_byte = [&](uint8_t value) {
            canonical.push_back(value);
        };
        auto append_u32 = [&](uint32_t value) {
            // Canonical little-endian representation: stable across compiler
            // enum layouts, bool sizes and host endianness.
            for (unsigned shift = 0; shift < 32; shift += 8) {
                append_byte(static_cast<uint8_t>(
                    (value >> shift) & 0xffu));
            }
        };
        auto append_i32 = [&](int value) {
            append_u32(static_cast<uint32_t>(
                static_cast<int32_t>(value)));
        };
        auto append_bool = [&](bool value) {
            append_byte(value ? uint8_t{1} : uint8_t{0});
        };
        auto append_float = [&](float value) {
            static_assert(sizeof(float) == sizeof(uint32_t),
                          "NSOS checkpoint format requires float32");
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            append_u32(bits);
        };
        const ModelConfig& c = model->model_config();
        append_i32(c.architecture_schema_version);
        append_i32(c.num_layers); append_i32(c.d_model);
        append_i32(c.vocab_size);
        append_i32(c.n_heads); append_i32(c.n_kv_heads);
        append_i32(c.sliding_window);
        append_i32(c.attention_period); append_i32(c.attention_slot);
        append_i32(static_cast<int>(c.hybrid_composition));
        append_bool(c.force_mamba_last_layer);
        append_bool(c.faithful_attention_linears);
        append_float(c.hybrid_mamba_gate_init);
        append_float(c.hybrid_attention_gate_init);
        append_float(c.hybrid_ffn_gate_init);
        append_float(c.rope_theta); append_i32(c.num_experts);
        append_i32(c.num_experts_per_token); append_bool(c.use_moe);
        append_i32(c.moe_period); append_i32(c.moe_slot);
        append_i32(c.moe_expert_hidden_dim); append_bool(c.use_ttt);
        append_i32(c.ttt_period); append_i32(c.ttt_slot);
        append_bool(c.use_chrass);
        append_float(c.chrass_density); append_u32(c.chrass_seed);
        append_float(c.logit_l2_beta);
        append_float(c.pantheon_vib_beta);
        append_bool(c.use_slender_embedding);
        append_bool(c.use_kan);
        append_bool(c.mamba_proper_ssm);
        append_bool(c.mamba_state_expansion);
        append_i32(c.mamba_d_state); append_i32(c.mamba_conv_kernel);
        append_bool(c.mamba2_faithful); append_i32(c.mamba_expand);
        append_i32(c.mamba_head_dim); append_i32(c.mamba_n_groups);
        append_bool(c.tie_word_embeddings);
        append_bool(c.use_gradient_checkpointing);
        append_float(c.dropout); append_i32(c.max_context_tokens);
        append_bool(c.use_exact_attention_training);
        // Conditional extension preserves every schema v1/v2 digest.
        if(c.architecture_schema_version >= 3) {
            append_bool(c.mamba3_enabled);
            append_i32(c.mamba3_schema_version);
            append_i32(c.mamba3_state_dim);
            append_bool(c.mamba3_mimo);
            append_i32(c.mamba3_mimo_rank);
            append_bool(c.mamba3_outproj_norm);
            append_float(c.mamba3_rope_fraction);
            append_float(c.mamba3_norm_eps);
            append_float(c.mamba3_a_floor);
        }
        return integrity::sha256_hex(canonical.data(), canonical.size());
    }

    // SHA-256 of the exact v4 bytes preceding the checkpoint integrity
    // trailer. This is intentionally the same byte stream emitted by save(),
    // including the absolute-name trailer. It lets the Trainer prove that a
    // sidecar is being paired with the live weights that produced it instead
    // of merely proving that some checkpoint file exists at the supplied
    // path.
    static std::string live_checkpoint_payload_digest(
        JambaModel* model) {
        if (!model) {
            throw std::invalid_argument(
                "Cannot digest a null runtime model");
        }
        static_assert(
            NSOS_MODEL_VERSION == 4,
            "Update live_checkpoint_payload_digest when the model format "
            "changes");
        static_assert(sizeof(uint32_t) == 4);
        static_assert(sizeof(int32_t) == 4);
        static_assert(sizeof(float) == 4);

        return integrity::sha256_hex_stream(
            [&](const integrity::Sha256Sink& sink) {
                const uint32_t magic = NSOS_MODEL_MAGIC;
                const uint32_t version = NSOS_MODEL_VERSION;
                const uint32_t fingerprint =
                    architecture_fingerprint(model);
                sink(&magic, sizeof(magic));
                sink(&version, sizeof(version));
                sink(&fingerprint, sizeof(fingerprint));

                const std::string config_digest =
                    configuration_digest(model);
                if (config_digest.size() != 64) {
                    throw std::logic_error(
                        "Internal SHA-256 configuration digest has invalid "
                        "length");
                }
                sink(config_digest.data(), config_digest.size());

                const auto params = model->parameters();
                if (params.size() >
                    static_cast<size_t>(
                        std::numeric_limits<uint32_t>::max())) {
                    throw std::overflow_error(
                        "Model has too many parameters for checkpoint v4");
                }
                const uint32_t count =
                    static_cast<uint32_t>(params.size());
                sink(&count, sizeof(count));

                std::unordered_set<std::string> stable_names;
                stable_names.reserve(params.size() * 2);
                for (const Parameter* parameter : params) {
                    if (!parameter || parameter->name.empty()) {
                        throw std::runtime_error(
                            "Cannot digest checkpoint: parameter has no "
                            "absolute name");
                    }
                    const std::string& stable_name = parameter->name;
                    if (!stable_names.insert(stable_name).second) {
                        throw std::runtime_error(
                            "Cannot digest checkpoint: duplicate absolute "
                            "parameter name '" + stable_name + "'");
                    }
                    if (stable_name.size() >
                        static_cast<size_t>(
                            std::numeric_limits<uint32_t>::max())) {
                        throw std::overflow_error(
                            "Checkpoint parameter name is too long");
                    }
                    const uint32_t name_len =
                        static_cast<uint32_t>(stable_name.size());
                    sink(&name_len, sizeof(name_len));
                    sink(stable_name.data(), stable_name.size());

                    if (parameter->data.shape.size() >
                        static_cast<size_t>(
                            std::numeric_limits<uint32_t>::max())) {
                        throw std::overflow_error(
                            "Checkpoint tensor rank is too large");
                    }
                    const uint32_t rank = static_cast<uint32_t>(
                        parameter->data.shape.size());
                    sink(&rank, sizeof(rank));
                    for (int dimension :
                         parameter->data.shape.dims) {
                        const int32_t encoded_dimension =
                            static_cast<int32_t>(dimension);
                        if (static_cast<int>(encoded_dimension) !=
                            dimension) {
                            throw std::overflow_error(
                                "Checkpoint tensor dimension is out of "
                                "int32 range");
                        }
                        sink(&encoded_dimension,
                             sizeof(encoded_dimension));
                    }

                    const Tensor host = parameter->data.cpu();
                    const uint64_t byte_count =
                        static_cast<uint64_t>(host.size) *
                        sizeof(float);
                    if (byte_count >
                        std::numeric_limits<uint32_t>::max()) {
                        throw std::overflow_error(
                            "Checkpoint tensor is too large for v4");
                    }
                    const float* values = host.data();
                    for (int64_t element = 0;
                         element < host.size; ++element) {
                        if (!std::isfinite(values[element])) {
                            throw std::runtime_error(
                                "Cannot digest checkpoint: parameter '" +
                                stable_name +
                                "' contains a non-finite value at element " +
                                std::to_string(element));
                        }
                    }
                    const uint32_t bytes =
                        static_cast<uint32_t>(byte_count);
                    sink(&bytes, sizeof(bytes));
                    sink(values, static_cast<size_t>(bytes));
                }

                const uint32_t name_trailer_magic =
                    0x4E534E32u;  // NSN2
                sink(&name_trailer_magic,
                     sizeof(name_trailer_magic));
                sink(&count, sizeof(count));
                for (const Parameter* parameter : params) {
                    const uint32_t name_len =
                        static_cast<uint32_t>(
                            parameter->name.size());
                    sink(&name_len, sizeof(name_len));
                    sink(parameter->name.data(),
                         parameter->name.size());
                }
            });
    }

    // Verifies both the checkpoint's own trailer and exact equality between
    // its v4 payload and the current runtime model. Legacy checkpoints must be
    // loaded and re-saved once before they can anchor a new optimizer sidecar;
    // accepting a format whose live-byte mapping is ambiguous would recreate
    // the torn model/optimizer-pair failure this guard is meant to prevent.
    static void verify_checkpoint_matches_live_model(
        JambaModel* model,
        const std::filesystem::path& checkpoint) {
        if (!model) {
            throw std::invalid_argument(
                "Cannot verify a checkpoint against a null runtime model");
        }
        constexpr uint64_t kIntegrityTrailerBytes =
            sizeof(uint32_t) + sizeof(uint64_t) + 64u;
        const uint64_t file_bytes =
            static_cast<uint64_t>(
                std::filesystem::file_size(checkpoint));
        if (file_bytes <
            sizeof(uint32_t) * 2u + kIntegrityTrailerBytes) {
            throw std::runtime_error(
                "Model checkpoint is too small for v4 integrity metadata");
        }

        std::ifstream input(checkpoint, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Cannot open model checkpoint for live-state verification");
        }
        uint32_t magic = 0;
        uint32_t version = 0;
        input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!input || magic != NSOS_MODEL_MAGIC) {
            throw std::runtime_error(
                "Model checkpoint has an invalid header");
        }
        if (version != NSOS_MODEL_VERSION ||
            version != 4u) {
            throw std::runtime_error(
                "Optimizer sidecars require a current v4 model checkpoint; "
                "load and re-save this legacy checkpoint first");
        }

        input.seekg(static_cast<std::streamoff>(
            file_bytes - kIntegrityTrailerBytes));
        uint32_t integrity_magic = 0;
        uint64_t payload_bytes = 0;
        std::string expected_payload_digest(64, '\0');
        input.read(reinterpret_cast<char*>(&integrity_magic),
                   sizeof(integrity_magic));
        input.read(reinterpret_cast<char*>(&payload_bytes),
                   sizeof(payload_bytes));
        input.read(expected_payload_digest.data(),
                   static_cast<std::streamsize>(
                       expected_payload_digest.size()));
        if (!input || integrity_magic != 0x4E534934u ||
            payload_bytes !=
                file_bytes - kIntegrityTrailerBytes) {
            throw std::runtime_error(
                "Model checkpoint v4 integrity trailer is corrupt");
        }
        const std::string actual_payload_digest =
            integrity::sha256_file_prefix(
                checkpoint, payload_bytes);
        if (actual_payload_digest !=
            expected_payload_digest) {
            throw std::runtime_error(
                "Model checkpoint SHA-256 integrity mismatch");
        }
        if (actual_payload_digest !=
            live_checkpoint_payload_digest(model)) {
            throw std::runtime_error(
                "Model checkpoint does not represent the current live "
                "weights; save a fresh model checkpoint before its optimizer "
                "sidecar");
        }
    }

    // Exact digest algorithm written by checkpoint v3, retained so v3 packs
    // remain loadable after v4 added explicit hybrid-composition fields.
    static uint64_t configuration_digest_v3(const JambaModel* model) {
        constexpr uint64_t kOffset = 1469598103934665603ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        uint64_t hash = kOffset;
        auto append = [&](const auto& value) {
            const auto* bytes =
                reinterpret_cast<const unsigned char*>(&value);
            for (size_t i = 0; i < sizeof(value); ++i) {
                hash ^= bytes[i];
                hash *= kPrime;
            }
        };
        const ModelConfig& c = model->model_config();
        append(c.num_layers); append(c.d_model); append(c.vocab_size);
        append(c.n_heads); append(c.n_kv_heads); append(c.sliding_window);
        append(c.attention_period); append(c.attention_slot);
        append(c.rope_theta); append(c.num_experts);
        append(c.num_experts_per_token); append(c.use_moe);
        append(c.moe_period); append(c.moe_slot);
        append(c.moe_expert_hidden_dim); append(c.use_ttt);
        append(c.ttt_period); append(c.ttt_slot); append(c.use_chrass);
        append(c.chrass_density); append(c.chrass_seed);
        append(c.logit_l2_beta); append(c.pantheon_vib_beta);
        append(c.use_slender_embedding); append(c.use_kan);
        append(c.mamba_proper_ssm); append(c.mamba_state_expansion);
        append(c.mamba_d_state); append(c.mamba_conv_kernel);
        append(c.mamba2_faithful); append(c.mamba_expand);
        append(c.mamba_head_dim); append(c.mamba_n_groups);
        append(c.tie_word_embeddings); append(c.use_gradient_checkpointing);
        append(c.dropout); append(c.max_context_tokens);
        append(c.use_exact_attention_training);
        return hash;
    }

private:
    // Exact legacy v3 checksum. Retained only for reading existing v3 files;
    // v4 writes SHA-256 for both config and payload integrity.
    static uint64_t fnv1a_file_prefix(const std::filesystem::path& path,
                                      uint64_t bytes_to_hash) {
        constexpr uint64_t kOffset = 1469598103934665603ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot hash checkpoint");
        uint64_t hash = kOffset;
        uint64_t consumed = 0;
        std::array<char, 1 << 16> buffer{};
        while (consumed < bytes_to_hash) {
            const uint64_t remaining = bytes_to_hash - consumed;
            const std::streamsize wanted = static_cast<std::streamsize>(
                (std::min)(remaining,
                           static_cast<uint64_t>(buffer.size())));
            input.read(buffer.data(), wanted);
            if (input.gcount() != wanted) {
                throw std::runtime_error(
                    "Checkpoint truncated while computing legacy checksum");
            }
            for (std::streamsize i = 0; i < wanted; ++i) {
                hash ^= static_cast<unsigned char>(
                    buffer[static_cast<size_t>(i)]);
                hash *= kPrime;
            }
            consumed += static_cast<uint64_t>(wanted);
        }
        return hash;
    }

public:
    static void save(JambaModel* model, const std::string& filename) {
        std::ofstream fs(filename, std::ios::binary | std::ios::trunc);
        if (!fs) {
            throw std::runtime_error(
                "Cannot open model checkpoint for writing: " + filename);
        }

        uint32_t magic = NSOS_MODEL_MAGIC;
        uint32_t version = NSOS_MODEL_VERSION;
        fs.write((char*)&magic, 4);
        fs.write((char*)&version, 4);
        // v2+: architecture fingerprint right after the version.
        uint32_t fingerprint = architecture_fingerprint(model);
        fs.write((char*)&fingerprint, 4);
        const std::string config_digest = configuration_digest(model);
        if (config_digest.size() != 64) {
            throw std::logic_error(
                "Internal SHA-256 configuration digest has invalid length");
        }
        fs.write(config_digest.data(),
                 static_cast<std::streamsize>(config_digest.size()));

        auto params = model->parameters();
        uint32_t count = (uint32_t)params.size();
        fs.write((char*)&count, 4);
        std::unordered_set<std::string> stable_names;
        stable_names.reserve(params.size() * 2);

        for (auto* p : params) {
            if (!p || p->name.empty()) {
                throw std::runtime_error(
                    "Cannot save checkpoint: parameter has no absolute name");
            }
            const std::string stable_name = p->name;
            if (!stable_names.insert(stable_name).second) {
                throw std::runtime_error(
                    "Cannot save checkpoint: duplicate absolute parameter "
                    "name '" + stable_name + "'");
            }
            // Write Name
            uint32_t name_len = (uint32_t)stable_name.size();
            fs.write((char*)&name_len, 4);
            fs.write(stable_name.c_str(), name_len);

            // Write Shape
            uint32_t rank = (uint32_t)p->data.shape.size();
            fs.write((char*)&rank, 4);
            for (int d : p->data.shape.dims) {
                int32_t val = d;
                fs.write((char*)&val, 4);
            }

            // Write Data
            Tensor cpu_t = p->data.cpu();
            const float* values = cpu_t.data();
            for (int64_t element = 0; element < cpu_t.size; ++element) {
                if (!std::isfinite(values[element])) {
                    throw std::runtime_error(
                        "Cannot save checkpoint: parameter '" + stable_name +
                        "' contains a non-finite value at element " +
                        std::to_string(element));
                }
            }
            const uint64_t bytes64 =
                static_cast<uint64_t>(cpu_t.size) * sizeof(float);
            if (bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Checkpoint tensor too large for serializer format");
            }
            uint32_t bytes = static_cast<uint32_t>(bytes64);
            fs.write((char*)&bytes, 4);
            fs.write((char*)cpu_t.data(), bytes);
        }

        // (auditoria #14) Trailer OPCIONAL v2: nomes absolutos por parâmetro.
        // A identidade primária continua base_name#ocorrência (ordem de
        // parameters()) — inserir/remover um módulo desloca ocorrências e o
        // load fica silenciosamente errado; o trailer dá ao loader uma
        // verificação secundária.  Arquivos antigos (sem trailer) carregam
        // exatamente como antes: o loader trata EOF aqui como formato v1.
        const uint32_t trailer_magic = 0x4E534E32u;  // 'NSN2'
        fs.write((char*)&trailer_magic, 4);
        fs.write((char*)&count, 4);
        for (auto* p : params) {
            const std::string& abs_name = p->name;
            uint32_t len = (uint32_t)abs_name.size();
            fs.write((char*)&len, 4);
            fs.write(abs_name.c_str(), len);
        }
        fs.flush();
        if (!fs) {
            throw std::runtime_error("Checkpoint write/flush failed");
        }
        fs.close();
        if (!fs) {
            throw std::runtime_error("Checkpoint close failed");
        }

        const uint64_t payload_bytes =
            static_cast<uint64_t>(std::filesystem::file_size(filename));
        const std::string payload_hash =
            integrity::sha256_file_prefix(filename, payload_bytes);
        if (payload_hash.size() != 64) {
            throw std::logic_error(
                "Internal SHA-256 checkpoint digest has invalid length");
        }
        std::ofstream append(filename, std::ios::binary | std::ios::app);
        if (!append) {
            throw std::runtime_error(
                "Cannot reopen checkpoint for integrity trailer");
        }
        const uint32_t integrity_magic = 0x4E534934u;  // NSI4
        append.write(reinterpret_cast<const char*>(&integrity_magic), 4);
        append.write(reinterpret_cast<const char*>(&payload_bytes), 8);
        append.write(payload_hash.data(),
                     static_cast<std::streamsize>(payload_hash.size()));
        append.flush();
        if (!append) {
            throw std::runtime_error(
                "Checkpoint integrity trailer write failed");
        }
        append.close();
        if (!append) {
            throw std::runtime_error(
                "Checkpoint integrity trailer close failed");
        }
    }

    static void load(JambaModel* model, const std::string& filename, bool strict = true) {
        std::ifstream fs(filename, std::ios::binary);
        if (!fs) throw std::runtime_error("Cannot open file for reading");

        uint32_t magic, version;
        fs.read((char*)&magic, 4);
        fs.read((char*)&version, 4);

        if (!fs) throw std::runtime_error("Checkpoint truncated in header");
        if (magic != NSOS_MODEL_MAGIC) throw std::runtime_error("Security: Invalid Magic Number");
        if (version == 0 || version > NSOS_MODEL_VERSION) {
            throw std::runtime_error("Security: Version Mismatch");
        }
        const uint64_t file_bytes =
            static_cast<uint64_t>(std::filesystem::file_size(filename));

        uint64_t validated_payload_bytes = 0;
        if (version >= 3) {
            const uint64_t integrity_bytes = version >= 4 ? 76u : 20u;
            if (file_bytes < integrity_bytes + 16) {
                throw std::runtime_error(
                    "Checkpoint is too small for its integrity trailer");
            }
            fs.seekg(static_cast<std::streamoff>(
                file_bytes - integrity_bytes));
            uint32_t integrity_magic = 0;
            uint64_t payload_bytes = 0;
            fs.read(reinterpret_cast<char*>(&integrity_magic), 4);
            fs.read(reinterpret_cast<char*>(&payload_bytes), 8);
            if (version >= 4) {
                std::string expected_hash(64, '\0');
                fs.read(expected_hash.data(),
                        static_cast<std::streamsize>(
                            expected_hash.size()));
                if (!fs || integrity_magic != 0x4E534934u ||
                    payload_bytes != file_bytes - integrity_bytes) {
                    throw std::runtime_error(
                        "Checkpoint SHA-256 trailer is missing or corrupt");
                }
                if (integrity::sha256_file_prefix(
                        filename, payload_bytes) != expected_hash) {
                    throw std::runtime_error(
                        "Checkpoint SHA-256 integrity mismatch");
                }
            } else {
                uint64_t expected_hash = 0;
                fs.read(reinterpret_cast<char*>(&expected_hash), 8);
                if (!fs || integrity_magic != 0x4E534933u ||
                    payload_bytes != file_bytes - integrity_bytes) {
                    throw std::runtime_error(
                        "Checkpoint v3 integrity trailer is missing or corrupt");
                }
                if (fnv1a_file_prefix(filename, payload_bytes) !=
                    expected_hash) {
                    throw std::runtime_error(
                        "Checkpoint v3 integrity checksum mismatch");
                }
            }
            validated_payload_bytes = payload_bytes;
            fs.clear();
            fs.seekg(8, std::ios::beg);
        }

        // ── v2 fingerprint check (actionable errors instead of cryptic
        // "parameter not found weight#N" when architectures diverge) ─────────
        const uint32_t runtime_fp = architecture_fingerprint(model);
        if(model->model_config().mamba3_enabled && version<4) throw std::runtime_error("Legacy checkpoint cannot be loaded into Mamba3");
        bool checkpoint_a_is_rate = false;  // v1: A stored as decay RATE
        if (version >= 2) {
            uint32_t ckpt_fp = 0;
            fs.read((char*)&ckpt_fp, 4);
            if (!fs) throw std::runtime_error("Checkpoint truncado lendo fingerprint");
            auto flag_mismatch = [&](uint32_t bit, const char* cfg_name) {
                if ((ckpt_fp & bit) != (runtime_fp & bit)) {
                    const bool ckpt_on = (ckpt_fp & bit) != 0;
                    throw std::runtime_error(
                        std::string("Checkpoint/architecture mismatch: checkpoint was saved with ") +
                        cfg_name + "=" + (ckpt_on ? "true" : "false") +
                        " but the runtime model was constructed with " + cfg_name + "=" +
                        (ckpt_on ? "false" : "true") +
                        ".  Construct the model with ModelConfig::" + cfg_name +
                        " matching the checkpoint and retry.");
                }
            };
            flag_mismatch(NSOS_FP_MAMBA_PROPER, "mamba_proper_ssm");
            flag_mismatch(NSOS_FP_STATE_EXPANSION, "mamba_state_expansion");
            flag_mismatch(NSOS_FP_TIE_EMBEDDINGS, "tie_word_embeddings");
            flag_mismatch(NSOS_FP_MAMBA2_FAITHFUL, "mamba2_faithful");
            flag_mismatch(NSOS_FP_MAMBA3, "mamba3_enabled");
            if(model->model_config().mamba3_enabled && version<4) throw std::runtime_error("Mamba3 requires checkpoint v4 architecture integrity");
            if (version >= 4) {
                flag_mismatch(NSOS_FP_HYBRID_PARALLEL,
                              "hybrid_composition");
                flag_mismatch(NSOS_FP_EXACT_ATTN_LINEAR,
                              "faithful_attention_linears");
                flag_mismatch(NSOS_FP_FORCE_MAMBA_LAST,
                              "force_mamba_last_layer");
            }
            checkpoint_a_is_rate = (ckpt_fp & NSOS_FP_A_LOG_DOMAIN) == 0;
            if (version >= 3) {
                if (version >= 4) {
                    std::string checkpoint_config_digest(64, '\0');
                    fs.read(
                        checkpoint_config_digest.data(),
                        static_cast<std::streamsize>(
                            checkpoint_config_digest.size()));
                    if (!fs) {
                        throw std::runtime_error(
                            "Checkpoint truncated reading SHA-256 "
                            "configuration digest");
                    }
                    if ((strict || model->model_config().mamba3_enabled) &&
                        checkpoint_config_digest !=
                            configuration_digest(model)) {
                        throw std::runtime_error(
                            "Checkpoint/architecture mismatch: complete "
                            "ModelConfig SHA-256 digest differs");
                    }
                } else {
                    uint64_t checkpoint_config_digest = 0;
                    fs.read(
                        reinterpret_cast<char*>(
                            &checkpoint_config_digest),
                        8);
                    if (!fs) {
                        throw std::runtime_error(
                            "Checkpoint truncated reading legacy "
                            "configuration digest");
                    }
                    if ((strict || model->model_config().mamba3_enabled) &&
                        checkpoint_config_digest !=
                            configuration_digest_v3(model)) {
                        throw std::runtime_error(
                            "Checkpoint/architecture mismatch: complete "
                            "legacy ModelConfig digest differs");
                    }
                }
            }
        } else {
            // v1 predates the fingerprint AND the A log-domain reparameterization
            // (N1): its Mamba `A` values are decay RATES in (0, 1].  Loading them
            // unconverted into the current code (which reads A as A_log and
            // applies exp) silently corrupts every decay by orders of magnitude.
            // We migrate exactly below (A_log = log(max(A, 1e-3)) reproduces the
            // old effective decay bit-for-bit in the recurrence).
            checkpoint_a_is_rate = true;
            std::cerr << "[ModelSerializer] AVISO: checkpoint v1 (pre-fingerprint). "
                         "Migrando Mamba A de rate->log-domain (exato). Se o load "
                         "falhar com 'parameter not found', o checkpoint foi treinado "
                         "com outra arquitetura — construa o modelo com os flags "
                         "ModelConfig correspondentes (mamba_proper_ssm=false / "
                         "tie_word_embeddings=false sao os defaults da epoca v1).\n";
        }

        uint32_t count;
        fs.read((char*)&count, 4);
        if (!fs) {
            throw std::runtime_error(
                "Checkpoint truncated reading parameter count");
        }
        if (count > 100000) throw std::runtime_error("Security: Parameter count exceeds limit");

        auto params = model->parameters();
        std::unordered_map<std::string, Parameter*> params_by_name;
        params_by_name.reserve(params.size() * 2);
        std::unordered_map<std::string, size_t> runtime_name_counts;
        std::unordered_map<Parameter*, std::string> canonical_name_for_param;
        canonical_name_for_param.reserve(params.size());
        std::vector<std::string> expected_runtime_names;
        expected_runtime_names.reserve(params.size());
        std::unordered_set<std::string> canonical_runtime_names;
        canonical_runtime_names.reserve(params.size() * 2);
        for (auto* p : params) {
            if (!p) {
                throw std::runtime_error(
                    "Runtime model contains a null parameter");
            }
            const std::string identity = !p->base_name.empty() ? p->base_name : p->name;
            const size_t occurrence = runtime_name_counts[identity]++;
            const std::string canonical_name =
                version >= 4 ? p->name
                             : identity + "#" + std::to_string(occurrence);
            if (canonical_name.empty() ||
                !canonical_runtime_names.insert(canonical_name).second) {
                throw std::runtime_error(
                    "Runtime model contains a missing or duplicate parameter "
                    "identity: '" + canonical_name + "'");
            }
            expected_runtime_names.push_back(canonical_name);
            canonical_name_for_param.emplace(p, canonical_name);
            params_by_name.emplace(canonical_name, p);
            if (version < 4) {
                // Legacy bodies were identified by base_name#occurrence.
                // Aliases are intentionally unavailable to v4, otherwise a
                // crafted partial checkpoint could bind an ambiguous leaf
                // name to whichever runtime parameter was registered first.
                if (!p->name.empty()) {
                    params_by_name.emplace(p->name, p);
                }
                if (!p->base_name.empty()) {
                    params_by_name.emplace(p->base_name, p);
                }
                // Historical registries prefixed an already-prefixed
                // embedding leaf and emitted "embedding.embedding.weight".
                // Keep this single, explicit migration alias for v1-v3 only;
                // v4 requires the canonical absolute "embedding.weight".
                if (p->name == "embedding.weight") {
                    params_by_name.emplace("embedding.embedding.weight", p);
                }
            }
        }
        std::unordered_map<std::string, bool> loaded_runtime_names;
        loaded_runtime_names.reserve(expected_runtime_names.size());
        uint32_t loaded_count = 0;
        uint32_t skipped_missing = 0;
        uint32_t skipped_shape = 0;
        struct PendingParameterLoad {
            Parameter* parameter = nullptr;
            std::vector<float> values;
        };
        std::vector<PendingParameterLoad> pending_loads;
        pending_loads.reserve(params.size());
        std::unordered_set<std::string> checkpoint_names;
        checkpoint_names.reserve(static_cast<size_t>(count) * 2);
        std::unordered_set<Parameter*> matched_parameters;
        matched_parameters.reserve(params.size() * 2);

        auto shape_to_string = [](const std::vector<int>& shape) {
            std::ostringstream out;
            out << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) out << ",";
                out << shape[i];
            }
            out << "]";
            return out.str();
        };

        // Endurecimento contra ARQUIVO NÃO CONFIÁVEL (model pack pode vir de
        // terceiros): cada leitura estrutural é conferida em fs ANTES de usar o
        // valor, e total_elements é acumulado com guarda de overflow + teto, em
        // vez de multiplicar size_t cru (rank 8 × dim 1e9 = 1e72 estoura size_t
        // por wraparound silencioso, podendo casar com um `bytes` forjado e
        // alocar/ler um buffer inconsistente).  Teto idêntico ao limite int do
        // tensor (checked_tensor_size).
        constexpr size_t kMaxElements =
            static_cast<size_t>(std::numeric_limits<int>::max());
        const uint64_t payload_limit =
            version >= 3 ? validated_payload_bytes : file_bytes;
        auto require_payload_bytes = [&](uint64_t count,
                                         const char* what) {
            const std::streampos position = fs.tellg();
            if (position < 0) {
                throw std::runtime_error(
                    std::string("Checkpoint stream position invalid before ") +
                    what);
            }
            const uint64_t offset = static_cast<uint64_t>(position);
            if (offset > payload_limit ||
                count > payload_limit - offset) {
                throw std::runtime_error(
                    std::string("Checkpoint truncated reading ") + what);
            }
        };
        auto read_field = [&](void* dst, std::streamsize n, const char* what) {
            if (n < 0) {
                throw std::logic_error(
                    "Negative checkpoint field size");
            }
            require_payload_bytes(static_cast<uint64_t>(n), what);
            fs.read(reinterpret_cast<char*>(dst), n);
            if (!fs) {
                throw std::runtime_error(std::string("Checkpoint truncado lendo ") + what);
            }
        };
        auto skip_field = [&](uint32_t n, const char* what) {
            require_payload_bytes(static_cast<uint64_t>(n), what);
            fs.seekg(static_cast<std::streamoff>(n), std::ios::cur);
            if (!fs) {
                throw std::runtime_error(
                    std::string("Checkpoint truncated skipping ") + what);
            }
        };

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t name_len;
            read_field(&name_len, 4, "name_len");
            if (name_len > 1024) throw std::runtime_error("Security: Name too long");

            std::string name(name_len, ' ');
            if (name_len > 0) read_field(&name[0], name_len, "name");
            if (name.empty()) {
                throw std::runtime_error(
                    "Security: Empty checkpoint parameter name");
            }
            if (!checkpoint_names.insert(name).second) {
                throw std::runtime_error(
                    "Security: Duplicate checkpoint parameter name: " +
                    name);
            }

            uint32_t rank;
            read_field(&rank, 4, "rank");
            if (rank > 8) throw std::runtime_error("Security: Rank too high");

            std::vector<int> shape(rank);
            size_t total_elements = 1;
            for (uint32_t j = 0; j < rank; ++j) {
                int32_t val;
                read_field(&val, 4, "dim");
                if (val < 0 || val > 1000000000) throw std::runtime_error("Security: Dimension invalid");
                shape[j] = val;
                const size_t dimension = static_cast<size_t>(val);
                if (dimension != 0 &&
                    total_elements > kMaxElements / dimension) {
                    throw std::runtime_error("Security: Tensor element count exceeds limit");
                }
                total_elements *= dimension;
            }

            uint32_t bytes;
            read_field(&bytes, 4, "byte_count");

            // Valida byte count contra o shape (agora sem overflow possível no
            // produto: total_elements <= kMaxElements < 2^31).
            if (static_cast<size_t>(bytes) != total_elements * sizeof(float))
                throw std::runtime_error("Security: Payload size mismatch");

            const auto candidate_it = params_by_name.find(name);
            if (candidate_it == params_by_name.end()) {
                if (strict) {
                    throw std::runtime_error(
                        "Checkpoint parameter not found in runtime model: " +
                        name);
                }
                ++skipped_missing;
                skip_field(bytes, "unknown parameter payload");
                continue;
            }
            Parameter* candidate = candidate_it->second;
            if (candidate->data.shape.dims != shape) {
                if (strict) {
                    throw std::runtime_error(
                        "Checkpoint shape mismatch for parameter '" + name +
                        "': checkpoint=" + shape_to_string(shape) +
                        " runtime=" +
                        shape_to_string(candidate->data.shape.dims));
                }
                ++skipped_shape;
                skip_field(bytes, "shape-mismatched parameter payload");
                continue;
            }

            // Alocação segura: total_elements já limitado a < 2^31 elementos.
            std::vector<float> buffer(total_elements);
            if (bytes > 0) read_field(buffer.data(), bytes, "payload");

            // v1→v2 A-domain migration: Mamba's decay parameter (stable identity
            // "A#k"; the only Parameter whose base_name is exactly "A") was a
            // RATE in v1 and is A_log now.  A_log = log(max(A, 1e-3)) reproduces
            // the old effective decay exactly (legacy code clamped at 1e-3).
            if (checkpoint_a_is_rate && name.rfind("A#", 0) == 0) {
                for (float& value : buffer) {
                    value = std::log(std::max(value, 1e-3f));
                }
            }
            for (size_t element = 0; element < buffer.size(); ++element) {
                if (!std::isfinite(buffer[element])) {
                    throw std::runtime_error(
                        "Security: Non-finite checkpoint value in parameter '" +
                        name + "' at element " +
                        std::to_string(element));
                }
            }

            const auto it = params_by_name.find(name);
            if (it == params_by_name.end()) {
                if (strict) {
                    throw std::runtime_error(
                        "Checkpoint parameter not found in runtime model: " + name +
                        ".  Likely architecture-flag mismatch — e.g. the checkpoint was "
                        "trained WITHOUT weight tying (construct with "
                        "ModelConfig::tie_word_embeddings=false) or with the legacy "
                        "Mamba path (mamba_proper_ssm=false).");
                }
                ++skipped_missing;
                continue;
            }

            Parameter* p = it->second;
            if (p->data.shape.dims != shape) {
                if (strict) {
                    throw std::runtime_error("Checkpoint shape mismatch for parameter '" + name +
                                             "': checkpoint=" + shape_to_string(shape) +
                                             " runtime=" + shape_to_string(p->data.shape.dims));
                }
                ++skipped_shape;
                continue;
            }
            if (!matched_parameters.insert(p).second) {
                throw std::runtime_error(
                    "Security: Multiple checkpoint identities resolve to "
                    "the same runtime parameter '" + p->name + "'");
            }

            pending_loads.push_back(
                PendingParameterLoad{p, std::move(buffer)});
            ++loaded_count;
            const auto canonical_it = canonical_name_for_param.find(p);
            if (canonical_it != canonical_name_for_param.end()) {
                loaded_runtime_names[canonical_it->second] = true;
            }
        }

        // (auditoria #14) Trailer v2 (opcional): nomes absolutos gravados
        // pelo save.  Arquivo v1 termina aqui (EOF) — comportamento idêntico
        // ao antigo.  Com trailer presente, compara nome absoluto por posição
        // e AVISA em divergência (reordenação de módulos desloca as
        // ocorrências da identidade primária silenciosamente).
        {
            uint32_t trailer_magic = 0;
            const bool has_name_trailer =
                static_cast<bool>(fs.read((char*)&trailer_magic, 4));
            if (has_name_trailer && trailer_magic == 0x4E534E32u) {
                uint32_t tcount = 0;
                if (!fs.read((char*)&tcount, 4)) {
                    throw std::runtime_error(
                        "Checkpoint truncated in parameter-name trailer");
                }
                // Não confiar no campo: limita ao count já validado do corpo.
                if ((version >= 3 && tcount != count) || tcount > count) {
                    throw std::runtime_error(
                        "Checkpoint parameter-name trailer count mismatch");
                }
                size_t mismatches = 0;
                std::string first_mismatch;
                for (uint32_t i = 0; i < tcount; ++i) {
                    uint32_t len = 0;
                    if (!fs.read((char*)&len, 4)) {
                        throw std::runtime_error(
                            "Checkpoint truncated in parameter-name length");
                    }
                    if (len > 4096) {
                        throw std::runtime_error(
                            "Checkpoint absolute parameter name is too long");
                    }
                    std::string abs_name(len, ' ');
                    if (len > 0 &&
                        !fs.read(abs_name.data(),
                                 static_cast<std::streamsize>(len))) {
                        throw std::runtime_error(
                            "Checkpoint truncated in absolute parameter name");
                    }
                    std::string normalized_abs_name = abs_name;
                    if (version < 4 &&
                        normalized_abs_name == "embedding.embedding.weight") {
                        normalized_abs_name = "embedding.weight";
                    }
                    if (i < params.size() && params[i] &&
                        params[i]->name != normalized_abs_name) {
                        ++mismatches;
                        if (first_mismatch.empty()) {
                            first_mismatch = abs_name + " != " + params[i]->name;
                        }
                    }
                }
                if (mismatches > 0) {
                    if (version >= 3 && strict) {
                        throw std::runtime_error(
                            "Checkpoint absolute parameter names do not match "
                            "the runtime architecture: " + first_mismatch);
                    }
                    std::cerr << "[ModelSerializer] AVISO: " << mismatches
                              << " nomes absolutos divergem do runtime (1o: "
                              << first_mismatch
                              << ") — possivel reordenacao de modulos.\n";
                }
                if (version >= 3) {
                    const std::streampos payload_end = fs.tellg();
                    if (payload_end < 0 ||
                        static_cast<uint64_t>(payload_end) !=
                            validated_payload_bytes) {
                        throw std::runtime_error(
                            "Checkpoint has unexpected bytes before the "
                            "integrity trailer");
                    }
                }
            } else if (version >= 3) {
                throw std::runtime_error(
                    "Checkpoint v3 parameter-name trailer is missing");
            }
            fs.clear();
        }

        if (strict) {
            for (const auto& expected_name : expected_runtime_names) {
                if (loaded_runtime_names.find(expected_name) == loaded_runtime_names.end()) {
                    throw std::runtime_error(
                        "Runtime parameter missing from checkpoint: " + expected_name +
                        ".  Likely architecture-flag mismatch — construct the model "
                        "with the ModelConfig flags the checkpoint was trained with "
                        "(mamba_proper_ssm / mamba_state_expansion / "
                        "tie_word_embeddings) and retry.");
                }
            }
        } else {
            if (loaded_count == 0) {
                throw std::runtime_error(
                    "Partial checkpoint load matched zero runtime parameters");
            }
            if (skipped_missing > 0 || skipped_shape > 0) {
                std::cerr << "[ModelSerializer] Partial load matched " << loaded_count
                          << " parameters, skipped " << skipped_missing
                          << " missing and " << skipped_shape
                          << " shape-mismatched parameters.\n";
            }
        }

        struct PreparedParameterCommit {
            Parameter* parameter = nullptr;
            Tensor staged;
        };
        static_assert(
            std::is_nothrow_move_assignable_v<Tensor>,
            "Transactional checkpoint publication requires noexcept Tensor "
            "move assignment");
        std::vector<PreparedParameterCommit> prepared;
        prepared.reserve(pending_loads.size());
        for (auto& pending : pending_loads) {
#ifdef NSOS_ENABLE_TEST_HOOKS
            detail::model_checkpoint_stage_fault_point();
#endif
            Parameter* parameter = pending.parameter;
            Tensor staged = Tensor::uninitialized(
                parameter->data.shape.dims, parameter->data.device);
            copy_tensor_bytes(
                staged.raw_data(), staged.device,
                pending.values.data(), Device::CPU,
                pending.values.size() * sizeof(float));
            prepared.push_back(
                PreparedParameterCommit{parameter, std::move(staged)});
        }

        // Reset captured graphs/session state only after every target-device
        // allocation and transfer has succeeded. Publication below consists
        // exclusively of noexcept moves, so an OOM/copy error cannot leave a
        // prefix of the live model updated.
        model->reset_session();
        for (auto& item : prepared) {
            item.parameter->data = std::move(item.staged);
            item.parameter->mark_updated();
        }
    }
};

} // namespace nsos
