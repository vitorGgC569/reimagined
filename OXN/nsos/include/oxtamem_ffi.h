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

    bool write(const std::string& key, const std::vector<uint8_t>& value);
    std::vector<uint8_t> read_latest(const std::string& key) const;
    std::vector<std::vector<uint8_t>> recall(const std::string& key, size_t depth) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nsos
