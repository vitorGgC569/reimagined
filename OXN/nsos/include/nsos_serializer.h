#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
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
    static void save(JambaModel* model, const std::string& filename) {
        std::ofstream fs(filename, std::ios::binary);
        if (!fs) throw std::runtime_error("Cannot open file for writing");

        uint32_t magic = NSOS_MODEL_MAGIC;
        uint32_t version = NSOS_MODEL_VERSION;
        fs.write((char*)&magic, 4);
        fs.write((char*)&version, 4);

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
    }

    static void load(JambaModel* model, const std::string& filename, bool strict = true) {
        std::ifstream fs(filename, std::ios::binary);
        if (!fs) throw std::runtime_error("Cannot open file for reading");

        uint32_t magic, version;
        fs.read((char*)&magic, 4);
        fs.read((char*)&version, 4);

        if (magic != NSOS_MODEL_MAGIC) throw std::runtime_error("Security: Invalid Magic Number");
        if (version > NSOS_MODEL_VERSION) throw std::runtime_error("Security: Version Mismatch");

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

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t name_len;
            fs.read((char*)&name_len, 4);
            if (name_len > 1024) throw std::runtime_error("Security: Name too long");

            std::string name(name_len, ' ');
            fs.read(&name[0], name_len);

            uint32_t rank;
            fs.read((char*)&rank, 4);
            if (rank > 8) throw std::runtime_error("Security: Rank too high");

            std::vector<int> shape(rank);
            size_t total_elements = 1;
            for (uint32_t j = 0; j < rank; ++j) {
                int32_t val;
                fs.read((char*)&val, 4);
                if (val < 0 || val > 1000000000) throw std::runtime_error("Security: Dimension invalid");
                shape[j] = val;
                total_elements *= val;
            }

            uint32_t bytes;
            fs.read((char*)&bytes, 4);
            
            // Validate byte size against shape
            if (bytes != total_elements * sizeof(float)) 
                throw std::runtime_error("Security: Payload size mismatch");

            // Safe Allocation
            std::vector<float> buffer(total_elements);
            fs.read((char*)buffer.data(), bytes);
            if (!fs) {
                throw std::runtime_error("Checkpoint payload truncated while reading " + name);
            }

            const auto it = params_by_name.find(name);
            if (it == params_by_name.end()) {
                if (strict) {
                    throw std::runtime_error("Checkpoint parameter not found in runtime model: " + name);
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

        if (strict) {
            for (const auto& expected_name : expected_runtime_names) {
                if (loaded_runtime_names.find(expected_name) == loaded_runtime_names.end()) {
                    throw std::runtime_error(
                        "Runtime parameter missing from checkpoint: " + expected_name);
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
