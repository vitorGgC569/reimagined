#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include "jamba.h"
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

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

        for (auto* p : params) {
            // Write Name
            uint32_t name_len = (uint32_t)p->name.size();
            fs.write((char*)&name_len, 4);
            fs.write(p->name.c_str(), name_len);

            // Write Shape
            uint32_t rank = (uint32_t)p->data.shape.size();
            fs.write((char*)&rank, 4);
            for (int d : p->data.shape.dims) {
                int32_t val = d;
                fs.write((char*)&val, 4);
            }

            // Write Data
            Tensor cpu_t = p->data.cpu();
            uint32_t bytes = cpu_t.size * sizeof(float);
            fs.write((char*)&bytes, 4);
            fs.write((char*)cpu_t.data(), bytes);
        }
    }

    static void load(JambaModel* model, const std::string& filename) {
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
        std::unordered_map<std::string, Parameter*> p_map;
        for (auto* p : params) p_map[p->name] = p;

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

            if (p_map.count(name)) {
                Parameter* p = p_map[name];
                // Check shape match before copy
                if (p->data.size != (int)total_elements) 
                    std::cerr << "[Warning] Size mismatch for " << name << ", skipping load." << std::endl;
                else
                    p->data.copy_from(Tensor::from_blob(buffer.data(), shape, Device::CPU).to(p->data.device));
            }
        }
    }
};

} // namespace nsos
