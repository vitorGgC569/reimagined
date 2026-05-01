#include "../include/causal_memory_store.h"

#include <chrono>
#include <fstream>
#include <stdexcept>

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
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream output(file_path_, std::ios::binary | std::ios::app);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to append to causal store");
    }

    output.seekp(0, std::ios::end);
    const int64_t offset = static_cast<int64_t>(output.tellp());
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

    heads_[key] = offset;
}

std::optional<CausalMemoryStore::RecordMeta> CausalMemoryStore::read_record_meta(
    std::ifstream& input, int64_t offset, std::string* out_key) const {
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

    if (out_key) {
        out_key->resize(meta.key_size);
        input.read(out_key->data(), static_cast<std::streamsize>(meta.key_size));
        if (!input.good()) {
            return std::nullopt;
        }
    } else {
        input.seekg(static_cast<std::streamoff>(meta.key_size), std::ios::cur);
    }
    return meta;
}

std::vector<uint8_t> CausalMemoryStore::read_payload(std::ifstream& input, int64_t offset,
                                                     const RecordMeta& meta) const {
    input.clear();
    input.seekg(offset + static_cast<int64_t>(sizeof(uint32_t) + sizeof(uint32_t) +
                                             sizeof(uint64_t) + sizeof(int64_t) +
                                             sizeof(uint64_t) + meta.key_size));
    if (!input.good()) {
        return {};
    }

    std::vector<uint8_t> payload(static_cast<size_t>(meta.payload_size));
    if (!payload.empty()) {
        input.read(reinterpret_cast<char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
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

    while (true) {
        const int64_t offset = static_cast<int64_t>(input.tellg());
        if (offset < 0 || input.peek() == EOF) {
            break;
        }

        std::string key;
        const auto meta = read_record_meta(input, offset, &key);
        if (!meta.has_value()) {
            break;
        }

        heads_[key] = offset;
        input.seekg(static_cast<std::streamoff>(meta->payload_size), std::ios::cur);
        if (!input.good()) {
            break;
        }
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

    const auto meta = read_record_meta(input, it->second);
    if (!meta.has_value()) {
        return std::nullopt;
    }
    return read_payload(input, it->second, *meta);
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

    int64_t cursor = it->second;
    while (cursor >= 0 && history.size() < depth) {
        const auto meta = read_record_meta(input, cursor);
        if (!meta.has_value()) {
            break;
        }
        history.push_back(read_payload(input, cursor, *meta));
        cursor = meta->prev_offset;
    }
    return history;
}

bool CausalMemoryStore::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return heads_.empty();
}

} // namespace nsos
