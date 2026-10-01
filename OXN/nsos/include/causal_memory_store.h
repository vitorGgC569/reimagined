#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nsos {

class CausalMemoryStore {
public:
    explicit CausalMemoryStore(const std::string& file_path);
    ~CausalMemoryStore();

    void append(const std::string& key, const std::vector<uint8_t>& payload);
    std::optional<std::vector<uint8_t>> read_latest(const std::string& key) const;
    std::vector<std::vector<uint8_t>> read_history(const std::string& key, size_t depth) const;
    bool empty() const;

private:
    struct ProcessLock;
    struct RecordMeta {
        uint32_t key_size = 0;
        uint64_t payload_size = 0;
        int64_t prev_offset = -1;
        uint64_t timestamp_ms = 0;
        uint64_t checksum = 0;
        uint64_t header_bytes = 0;
        uint64_t record_bytes = 0;
        bool checksummed = false;
    };

    static constexpr uint32_t kLegacyMagic = 0x43535452; // CSTR v1
    static constexpr uint32_t kMagic = 0x43535432;       // CST2
    static constexpr uint32_t kCommitMagic = 0x434D4954; // CMIT
    static constexpr uint64_t kLegacyHeaderBytes =
        sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t) +
        sizeof(int64_t) + sizeof(uint64_t);
    static constexpr uint64_t kHeaderBytes = kLegacyHeaderBytes + sizeof(uint64_t);
    static constexpr uint64_t kFooterBytes = sizeof(uint32_t) + sizeof(uint64_t);
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
    std::unique_ptr<ProcessLock> process_lock_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, int64_t> heads_;
};

} // namespace nsos
