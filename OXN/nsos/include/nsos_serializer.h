#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include <algorithm>
#include <cmath>
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

    static void save(JambaModel* model, const std::string& filename) {
        std::ofstream fs(filename, std::ios::binary);
        if (!fs) throw std::runtime_error("Cannot open file for writing");

        uint32_t magic = NSOS_MODEL_MAGIC;
        uint32_t version = NSOS_MODEL_VERSION;
        fs.write((char*)&magic, 4);
        fs.write((char*)&version, 4);
        // v2: architecture fingerprint right after the version.
        uint32_t fingerprint = architecture_fingerprint(model);
        fs.write((char*)&fingerprint, 4);

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
    }

    static void load(JambaModel* model, const std::string& filename, bool strict = true) {
        std::ifstream fs(filename, std::ios::binary);
        if (!fs) throw std::runtime_error("Cannot open file for reading");

        uint32_t magic, version;
        fs.read((char*)&magic, 4);
        fs.read((char*)&version, 4);

        if (magic != NSOS_MODEL_MAGIC) throw std::runtime_error("Security: Invalid Magic Number");
        if (version > NSOS_MODEL_VERSION) throw std::runtime_error("Security: Version Mismatch");

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
                total_elements *= static_cast<size_t>(val);
                if (total_elements > kMaxElements) {
                    throw std::runtime_error("Security: Tensor element count exceeds limit");
                }
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

            p->data.copy_from(
                Tensor::from_blob(buffer.data(), shape, Device::CPU).to(p->data.device));
            p->mark_updated();
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
            if (fs.read((char*)&trailer_magic, 4) && trailer_magic == 0x4E534E32u) {
                uint32_t tcount = 0;
                fs.read((char*)&tcount, 4);
                // Não confiar no campo: limita ao count já validado do corpo.
                if (tcount > count) tcount = count;
                size_t mismatches = 0;
                std::string first_mismatch;
                for (uint32_t i = 0; i < tcount && fs; ++i) {
                    uint32_t len = 0;
                    if (!fs.read((char*)&len, 4)) break;
                    if (len > 4096) break;
                    std::string abs_name(len, ' ');
                    fs.read(&abs_name[0], len);
                    if (i < params.size() && params[i] && params[i]->name != abs_name) {
                        ++mismatches;
                        if (first_mismatch.empty()) {
                            first_mismatch = abs_name + " != " + params[i]->name;
                        }
                    }
                }
                if (mismatches > 0) {
                    std::cerr << "[ModelSerializer] AVISO: " << mismatches
                              << " nomes absolutos divergem do runtime (1o: "
                              << first_mismatch
                              << ") — possivel reordenacao de modulos.\n";
                }
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
    }
};

} // namespace nsos
