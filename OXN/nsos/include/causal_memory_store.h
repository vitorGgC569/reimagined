#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nsos {

class CausalMemoryStore {
public:
    explicit CausalMemoryStore(const std::string& file_path);

    void append(const std::string& key, const std::vector<uint8_t>& payload);
    std::optional<std::vector<uint8_t>> read_latest(const std::string& key) const;
    std::vector<std::vector<uint8_t>> read_history(const std::string& key, size_t depth) const;
    bool empty() const;

private:
    struct RecordMeta {
        uint32_t key_size = 0;
        uint64_t payload_size = 0;
        int64_t prev_offset = -1;
        uint64_t timestamp_ms = 0;
    };

    static constexpr uint32_t kMagic = 0x43535452; // CSTR
    static constexpr uint64_t kHeaderBytes =
        sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t) +
        sizeof(int64_t) + sizeof(uint64_t);
    static constexpr uint32_t kMaxKeyBytes = 1024;
    static constexpr uint64_t kMaxPayloadBytes = 16ull * 1024ull * 1024ull;
    static constexpr uint64_t kMaxHistoryBytes = 64ull * 1024ull * 1024ull;
    static constexpr size_t kMaxHistoryDepth = 1024;
    static constexpr uint64_t kMaxStoreBytes = 32ull * 1024ull * 1024ull * 1024ull;

    void rebuild_index();
    std::optional<RecordMeta> read_record_meta(std::ifstream& input, int64_t offset,
                                               uint64_t file_size,
                                               std::string* out_key = nullptr) const;
    std::optional<std::vector<uint8_t>> read_payload(
        std::ifstream& input, int64_t offset, uint64_t file_size,
        const RecordMeta& meta) const;

    std::filesystem::path file_path_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, int64_t> heads_;
};

} // namespace nsos
