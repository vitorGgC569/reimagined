#include "../include/causal_memory_store.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace nsos {

namespace {

uint64_t now_unix_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

} // namespace

CausalMemoryStore::CausalMemoryStore(const std::string& file_path) : file_path_(file_path) {
    if (!file_path_.parent_path().empty()) {
        std::filesystem::create_directories(file_path_.parent_path());
    }
    if (!std::filesystem::exists(file_path_)) {
        std::ofstream output(file_path_, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to create causal store");
        }
    }
    rebuild_index();
}

void CausalMemoryStore::append(const std::string& key, const std::vector<uint8_t>& payload) {
    if (key.empty() || key.size() > kMaxKeyBytes) {
        throw std::invalid_argument("Causal store key must be 1..1024 bytes");
    }
    if (payload.size() > kMaxPayloadBytes) {
        throw std::invalid_argument("Causal store payload exceeds 16 MiB");
    }
    const uint64_t record_bytes =
        kHeaderBytes + static_cast<uint64_t>(key.size()) +
        static_cast<uint64_t>(payload.size());
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream output(file_path_, std::ios::binary | std::ios::app);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to append to causal store");
    }

    output.seekp(0, std::ios::end);
    const std::streampos end_pos = output.tellp();
    if (end_pos < 0) {
        throw std::runtime_error("Failed to determine causal store size");
    }
    const uint64_t offset_u64 = static_cast<uint64_t>(end_pos);
    if (offset_u64 > kMaxStoreBytes ||
        record_bytes > kMaxStoreBytes - offset_u64 ||
        offset_u64 >
            static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
        throw std::runtime_error("Causal store size limit exceeded");
    }
    const int64_t offset = static_cast<int64_t>(offset_u64);
    const int64_t prev_offset =
        heads_.count(key) > 0 ? heads_.at(key) : static_cast<int64_t>(-1);
    const uint32_t key_size = static_cast<uint32_t>(key.size());
    const uint64_t payload_size = static_cast<uint64_t>(payload.size());
    const uint64_t timestamp_ms = now_unix_ms();

    output.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    output.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
    output.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
    output.write(reinterpret_cast<const char*>(&prev_offset), sizeof(prev_offset));
    output.write(reinterpret_cast<const char*>(&timestamp_ms), sizeof(timestamp_ms));
    output.write(key.data(), static_cast<std::streamsize>(key.size()));
    if (!payload.empty()) {
        output.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size()));
    }
    output.flush();
    if (!output.good()) {
        throw std::runtime_error("Failed to flush causal store record");
    }

    heads_[key] = offset;
}

std::optional<CausalMemoryStore::RecordMeta> CausalMemoryStore::read_record_meta(
    std::ifstream& input, int64_t offset, uint64_t file_size,
    std::string* out_key) const {
    if (offset < 0) {
        return std::nullopt;
    }
    const uint64_t offset_u64 = static_cast<uint64_t>(offset);
    if (offset_u64 > file_size || kHeaderBytes > file_size - offset_u64) {
        return std::nullopt;
    }
    input.clear();
    input.seekg(offset);
    if (!input.good()) {
        return std::nullopt;
    }

    uint32_t magic = 0;
    RecordMeta meta;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char*>(&meta.key_size), sizeof(meta.key_size));
    input.read(reinterpret_cast<char*>(&meta.payload_size), sizeof(meta.payload_size));
    input.read(reinterpret_cast<char*>(&meta.prev_offset), sizeof(meta.prev_offset));
    input.read(reinterpret_cast<char*>(&meta.timestamp_ms), sizeof(meta.timestamp_ms));
    if (!input.good() || magic != kMagic) {
        return std::nullopt;
    }
    if (meta.key_size == 0 || meta.key_size > kMaxKeyBytes ||
        meta.payload_size > kMaxPayloadBytes) {
        return std::nullopt;
    }
    const uint64_t body_bytes =
        static_cast<uint64_t>(meta.key_size) + meta.payload_size;
    if (body_bytes > file_size - offset_u64 - kHeaderBytes) {
        return std::nullopt;
    }
    if (meta.prev_offset < -1 ||
        (meta.prev_offset >= 0 && meta.prev_offset >= offset)) {
        return std::nullopt;
    }

    if (out_key) {
        out_key->resize(meta.key_size);
        input.read(out_key->data(), static_cast<std::streamsize>(meta.key_size));
        if (!input.good()) {
            return std::nullopt;
        }
    } else {
        input.seekg(static_cast<std::streamoff>(meta.key_size), std::ios::cur);
    }
    if (!input.good()) {
        return std::nullopt;
    }
    return meta;
}

std::optional<std::vector<uint8_t>> CausalMemoryStore::read_payload(
    std::ifstream& input, int64_t offset, uint64_t file_size,
    const RecordMeta& meta) const {
    if (offset < 0 || meta.payload_size > kMaxPayloadBytes) {
        return std::nullopt;
    }
    const uint64_t payload_offset =
        static_cast<uint64_t>(offset) + kHeaderBytes + meta.key_size;
    if (payload_offset > file_size ||
        meta.payload_size > file_size - payload_offset) {
        return std::nullopt;
    }
    input.clear();
    input.seekg(static_cast<std::streamoff>(payload_offset));
    if (!input.good()) {
        return std::nullopt;
    }

    std::vector<uint8_t> payload(static_cast<size_t>(meta.payload_size));
    if (!payload.empty()) {
        input.read(reinterpret_cast<char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
        if (input.gcount() != static_cast<std::streamsize>(payload.size())) {
            return std::nullopt;
        }
    }
    return payload;
}

void CausalMemoryStore::rebuild_index() {
    std::lock_guard<std::mutex> lock(mutex_);
    heads_.clear();

    std::ifstream input(file_path_, std::ios::binary);
    if (!input.is_open()) {
        return;
    }
    std::error_code size_error;
    const uint64_t file_size = std::filesystem::file_size(file_path_, size_error);
    if (size_error) {
        throw std::runtime_error("Failed to inspect causal store size");
    }
    if (file_size > kMaxStoreBytes) {
        throw std::runtime_error("Causal store exceeds configured size limit");
    }

    uint64_t cursor = 0;
    while (cursor < file_size) {
        if (cursor >
            static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
            throw std::runtime_error("Causal store offset exceeds supported range");
        }
        input.clear();
        input.seekg(static_cast<std::streamoff>(cursor));
        const int64_t offset = static_cast<int64_t>(input.tellg());
        if (offset < 0) {
            throw std::runtime_error("Failed to seek causal store record");
        }

        std::string key;
        const auto meta = read_record_meta(input, offset, file_size, &key);
        if (!meta.has_value()) {
            throw std::runtime_error("Corrupt causal store record at offset " +
                                     std::to_string(offset));
        }
        const int64_t expected_prev =
            heads_.count(key) > 0 ? heads_.at(key) : static_cast<int64_t>(-1);
        if (meta->prev_offset != expected_prev) {
            throw std::runtime_error("Invalid causal store lineage at offset " +
                                     std::to_string(offset));
        }

        heads_[key] = offset;
        cursor += kHeaderBytes + static_cast<uint64_t>(meta->key_size) +
                  meta->payload_size;
    }
}

std::optional<std::vector<uint8_t>> CausalMemoryStore::read_latest(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = heads_.find(key);
    if (it == heads_.end()) {
        return std::nullopt;
    }

    std::ifstream input(file_path_, std::ios::binary);
    if (!input.is_open()) {
        return std::nullopt;
    }
    std::error_code size_error;
    const uint64_t file_size = std::filesystem::file_size(file_path_, size_error);
    if (size_error || file_size > kMaxStoreBytes) {
        return std::nullopt;
    }

    std::string stored_key;
    const auto meta = read_record_meta(input, it->second, file_size, &stored_key);
    if (!meta.has_value() || stored_key != key) {
        return std::nullopt;
    }
    return read_payload(input, it->second, file_size, *meta);
}

std::vector<std::vector<uint8_t>> CausalMemoryStore::read_history(const std::string& key,
                                                                  size_t depth) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::vector<uint8_t>> history;
    const auto it = heads_.find(key);
    if (it == heads_.end() || depth == 0) {
        return history;
    }

    std::ifstream input(file_path_, std::ios::binary);
    if (!input.is_open()) {
        return history;
    }
    std::error_code size_error;
    const uint64_t file_size = std::filesystem::file_size(file_path_, size_error);
    if (size_error || file_size > kMaxStoreBytes) {
        return history;
    }

    int64_t cursor = it->second;
    uint64_t history_bytes = 0;
    std::unordered_set<int64_t> visited;
    const size_t bounded_depth = std::min(depth, kMaxHistoryDepth);
    while (cursor >= 0 && history.size() < bounded_depth) {
        if (!visited.insert(cursor).second) {
            break;
        }
        std::string stored_key;
        const auto meta = read_record_meta(input, cursor, file_size, &stored_key);
        if (!meta.has_value() || stored_key != key ||
            meta->payload_size > kMaxHistoryBytes - history_bytes) {
            break;
        }
        auto payload = read_payload(input, cursor, file_size, *meta);
        if (!payload.has_value()) {
            break;
        }
        history_bytes += meta->payload_size;
        history.push_back(std::move(*payload));
        cursor = meta->prev_offset;
    }
    return history;
}

bool CausalMemoryStore::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return heads_.empty();
}

} // namespace nsos
