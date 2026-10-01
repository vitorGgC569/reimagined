#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nsos {

class OxtaMemFFI {
public:
    OxtaMemFFI();
    ~OxtaMemFFI();

    OxtaMemFFI(const OxtaMemFFI&) = delete;
    OxtaMemFFI& operator=(const OxtaMemFFI&) = delete;

    bool load(const std::string& library_path = {});
    bool open(const std::string& store_path, uint64_t size_mb = 128);
    bool is_ready() const;
    std::string last_error() const;

    bool write(const std::string& key, const std::vector<uint8_t>& value);
    bool write_with_vector(const std::string& key,
                           const std::vector<uint8_t>& value,
                           const std::vector<float>& vector);
    std::vector<uint8_t> read_latest(const std::string& key) const;
    std::vector<std::vector<uint8_t>> recall(const std::string& key, size_t depth) const;
    std::vector<std::vector<uint8_t>> search_similar(
        const std::vector<float>& vector, size_t top_k) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nsos
