#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
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
#include <unordered_map>

namespace nsos {

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
        return fp;
    }

    static uint64_t configuration_digest(const JambaModel* model) {
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
    static uint64_t hash_file_prefix(const std::filesystem::path& path,
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
                    "Checkpoint truncated while computing integrity checksum");
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
        if (!fs) throw std::runtime_error("Cannot open file for writing");

        uint32_t magic = NSOS_MODEL_MAGIC;
        uint32_t version = NSOS_MODEL_VERSION;
        fs.write((char*)&magic, 4);
        fs.write((char*)&version, 4);
        // v2: architecture fingerprint right after the version.
        uint32_t fingerprint = architecture_fingerprint(model);
        fs.write((char*)&fingerprint, 4);
        const uint64_t config_digest = configuration_digest(model);
        fs.write((char*)&config_digest, 8);

        auto params = model->parameters();
        uint32_t count = (uint32_t)params.size();
        fs.write((char*)&count, 4);
        std::unordered_map<std::string, size_t> stable_name_counts;

        for (auto* p : params) {
            const std::string identity = !p->base_name.empty() ? p->base_name : p->name;
            const size_t occurrence = stable_name_counts[identity]++;
            const std::string stable_name = identity + "#" + std::to_string(occurrence);
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
        const uint64_t payload_hash =
            hash_file_prefix(filename, payload_bytes);
        std::ofstream append(filename, std::ios::binary | std::ios::app);
        if (!append) {
            throw std::runtime_error(
                "Cannot reopen checkpoint for integrity trailer");
        }
        const uint32_t integrity_magic = 0x4E534933u;  // NSI3
        append.write(reinterpret_cast<const char*>(&integrity_magic), 4);
        append.write(reinterpret_cast<const char*>(&payload_bytes), 8);
        append.write(reinterpret_cast<const char*>(&payload_hash), 8);
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
        if (version > NSOS_MODEL_VERSION) throw std::runtime_error("Security: Version Mismatch");

        uint64_t validated_payload_bytes = 0;
        if (version >= 3) {
            constexpr uint64_t kIntegrityBytes = 20;
            const uint64_t file_bytes =
                static_cast<uint64_t>(std::filesystem::file_size(filename));
            if (file_bytes < kIntegrityBytes + 16) {
                throw std::runtime_error(
                    "Checkpoint is too small for the v3 integrity trailer");
            }
            fs.seekg(static_cast<std::streamoff>(file_bytes - kIntegrityBytes));
            uint32_t integrity_magic = 0;
            uint64_t payload_bytes = 0;
            uint64_t expected_hash = 0;
            fs.read(reinterpret_cast<char*>(&integrity_magic), 4);
            fs.read(reinterpret_cast<char*>(&payload_bytes), 8);
            fs.read(reinterpret_cast<char*>(&expected_hash), 8);
            if (!fs || integrity_magic != 0x4E534933u ||
                payload_bytes != file_bytes - kIntegrityBytes) {
                throw std::runtime_error(
                    "Checkpoint integrity trailer is missing or corrupt");
            }
            if (hash_file_prefix(filename, payload_bytes) != expected_hash) {
                throw std::runtime_error(
                    "Checkpoint integrity checksum mismatch");
            }
            validated_payload_bytes = payload_bytes;
            fs.clear();
            fs.seekg(8, std::ios::beg);
        }

        // ── v2 fingerprint check (actionable errors instead of cryptic
        // "parameter not found weight#N" when architectures diverge) ─────────
        const uint32_t runtime_fp = architecture_fingerprint(model);
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
            checkpoint_a_is_rate = (ckpt_fp & NSOS_FP_A_LOG_DOMAIN) == 0;
            if (version >= 3) {
                uint64_t checkpoint_config_digest = 0;
                fs.read(reinterpret_cast<char*>(&checkpoint_config_digest), 8);
                if (!fs) {
                    throw std::runtime_error(
                        "Checkpoint truncated reading configuration digest");
                }
                if (strict &&
                    checkpoint_config_digest !=
                        configuration_digest(model)) {
                    throw std::runtime_error(
                        "Checkpoint/architecture mismatch: complete "
                        "ModelConfig digest differs");
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
        if (count > 100000) throw std::runtime_error("Security: Parameter count exceeds limit");

        auto params = model->parameters();
        std::unordered_map<std::string, Parameter*> params_by_name;
        params_by_name.reserve(params.size() * 2);
        std::unordered_map<std::string, size_t> runtime_name_counts;
        std::unordered_map<Parameter*, std::string> canonical_name_for_param;
        canonical_name_for_param.reserve(params.size());
        std::vector<std::string> expected_runtime_names;
        expected_runtime_names.reserve(params.size());
        for (auto* p : params) {
            if (!p) {
                continue;
            }
            const std::string identity = !p->base_name.empty() ? p->base_name : p->name;
            const size_t occurrence = runtime_name_counts[identity]++;
            const std::string canonical_name = identity + "#" + std::to_string(occurrence);
            expected_runtime_names.push_back(canonical_name);
            canonical_name_for_param.emplace(p, canonical_name);
            params_by_name.emplace(canonical_name, p);
            if (!p->name.empty()) {
                params_by_name.emplace(p->name, p);
            }
            if (!p->base_name.empty()) {
                params_by_name.emplace(p->base_name, p);
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
        auto read_field = [&](void* dst, std::streamsize n, const char* what) {
            fs.read(reinterpret_cast<char*>(dst), n);
            if (!fs) {
                throw std::runtime_error(std::string("Checkpoint truncado lendo ") + what);
            }
        };

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t name_len;
            read_field(&name_len, 4, "name_len");
            if (name_len > 1024) throw std::runtime_error("Security: Name too long");

            std::string name(name_len, ' ');
            if (name_len > 0) read_field(&name[0], name_len, "name");

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
                    if (i < params.size() && params[i] && params[i]->name != abs_name) {
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

        // Commit only after every structural and semantic validation passes.
        // Rejected checkpoints therefore cannot half-mutate a live model.
        for (auto& pending : pending_loads) {
            Parameter* parameter = pending.parameter;
            parameter->data.copy_from(
                Tensor::from_blob(pending.values.data(),
                                  parameter->data.shape.dims,
                                  Device::CPU)
                    .to(parameter->data.device));
            parameter->mark_updated();
        }
    }
};

} // namespace nsos
