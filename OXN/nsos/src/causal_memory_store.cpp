#include "../include/causal_memory_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace nsos {

namespace {

uint64_t now_unix_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

template <typename T>
void write_le(std::ostream& output, T value) {
    using U = std::make_unsigned_t<T>;
    U bits = static_cast<U>(value);
    std::array<char, sizeof(T)> bytes{};
    for (size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((bits >> (index * 8)) & U{0xff});
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
bool read_le(std::istream& input, T& value) {
    using U = std::make_unsigned_t<T>;
    std::array<unsigned char, sizeof(T)> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input) return false;
    U bits = 0;
    for (size_t index = 0; index < bytes.size(); ++index) {
        bits |= static_cast<U>(bytes[index]) << (index * 8);
    }
    value = static_cast<T>(bits);
    return true;
}

uint64_t fnv_update(uint64_t hash, const void* data, size_t bytes) {
    const auto* input = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= input[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename T>
uint64_t fnv_update_le(uint64_t hash, T value) {
    using U = std::make_unsigned_t<T>;
    U bits = static_cast<U>(value);
    for (size_t index = 0; index < sizeof(T); ++index) {
        const uint8_t byte = static_cast<uint8_t>((bits >> (index * 8)) & U{0xff});
        hash = fnv_update(hash, &byte, 1);
    }
    return hash;
}

uint64_t record_checksum(const std::string& key,
                         const std::vector<uint8_t>& payload,
                         int64_t prev_offset,
                         uint64_t timestamp_ms) {
    uint64_t hash = 1469598103934665603ull;
    hash = fnv_update_le(hash, static_cast<uint32_t>(key.size()));
    hash = fnv_update_le(hash, static_cast<uint64_t>(payload.size()));
    hash = fnv_update_le(hash, prev_offset);
    hash = fnv_update_le(hash, timestamp_ms);
    hash = fnv_update(hash, key.data(), key.size());
    return fnv_update(hash, payload.data(), payload.size());
}

void durable_sync(const std::filesystem::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE |
                                    FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open causal store for durable sync");
    }
    const BOOL ok = FlushFileBuffers(handle);
    CloseHandle(handle);
    if (!ok) throw std::runtime_error("Failed to durably sync causal store");
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error("Failed to open causal store for durable sync");
    }
    const int status = ::fsync(descriptor);
    ::close(descriptor);
    if (status != 0) throw std::runtime_error("Failed to durably sync causal store");
#endif
}

} // namespace

struct CausalMemoryStore::ProcessLock {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int descriptor = -1;
#endif

    explicit ProcessLock(const std::filesystem::path& store_path) {
        std::filesystem::path lock_path = store_path;
        lock_path += ".lock";
#ifdef _WIN32
        handle = CreateFileW(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                             nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error(
                "Causal store is already open by another process");
        }
#else
        descriptor = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0600);
        if (descriptor < 0 || ::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
            if (descriptor >= 0) ::close(descriptor);
            descriptor = -1;
            throw std::runtime_error(
                "Causal store is already open by another process");
        }
#endif
    }

    ~ProcessLock() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (descriptor >= 0) ::close(descriptor);
#endif
    }
};

CausalMemoryStore::CausalMemoryStore(const std::string& file_path) : file_path_(file_path) {
    if (!file_path_.parent_path().empty()) {
        std::filesystem::create_directories(file_path_.parent_path());
    }
    process_lock_ = std::make_unique<ProcessLock>(file_path_);
    if (!std::filesystem::exists(file_path_)) {
        std::ofstream output(file_path_, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to create causal store");
        }
    }
    rebuild_index();
}

CausalMemoryStore::~CausalMemoryStore() = default;

void CausalMemoryStore::append(const std::string& key, const std::vector<uint8_t>& payload) {
    if (key.empty() || key.size() > kMaxKeyBytes) {
        throw std::invalid_argument("Causal store key must be 1..1024 bytes");
    }
    if (payload.size() > kMaxPayloadBytes) {
        throw std::invalid_argument("Causal store payload exceeds 16 MiB");
    }
    const uint64_t record_bytes =
        kHeaderBytes + static_cast<uint64_t>(key.size()) +
        static_cast<uint64_t>(payload.size()) + kFooterBytes;
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
    const uint64_t checksum =
        record_checksum(key, payload, prev_offset, timestamp_ms);

    write_le(output, kMagic);
    write_le(output, key_size);
    write_le(output, payload_size);
    write_le(output, prev_offset);
    write_le(output, timestamp_ms);
    write_le(output, checksum);
    output.write(key.data(), static_cast<std::streamsize>(key.size()));
    if (!payload.empty()) {
        output.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size()));
    }
    write_le(output, kCommitMagic);
    write_le(output, record_bytes);
    output.flush();
    if (!output.good()) {
        throw std::runtime_error("Failed to flush causal store record");
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Failed to close causal store record");
    }
    durable_sync(file_path_);

    heads_[key] = offset;
}

std::optional<CausalMemoryStore::RecordMeta> CausalMemoryStore::read_record_meta(
    std::ifstream& input, int64_t offset, uint64_t file_size,
    std::string* out_key) const {
    if (offset < 0) {
        return std::nullopt;
    }
    const uint64_t offset_u64 = static_cast<uint64_t>(offset);
    if (offset_u64 > file_size || sizeof(uint32_t) > file_size - offset_u64) {
        return std::nullopt;
    }
    input.clear();
    input.seekg(offset);
    if (!input.good()) {
        return std::nullopt;
    }

    uint32_t magic = 0;
    RecordMeta meta;
    if (!read_le(input, magic) || (magic != kMagic && magic != kLegacyMagic)) {
        return std::nullopt;
    }
    meta.checksummed = magic == kMagic;
    meta.header_bytes = meta.checksummed ? kHeaderBytes : kLegacyHeaderBytes;
    if (meta.header_bytes > file_size - offset_u64 ||
        !read_le(input, meta.key_size) ||
        !read_le(input, meta.payload_size) ||
        !read_le(input, meta.prev_offset) ||
        !read_le(input, meta.timestamp_ms) ||
        (meta.checksummed && !read_le(input, meta.checksum))) {
        return std::nullopt;
    }
    if (meta.key_size == 0 || meta.key_size > kMaxKeyBytes ||
        meta.payload_size > kMaxPayloadBytes) {
        return std::nullopt;
    }
    const uint64_t body_bytes =
        static_cast<uint64_t>(meta.key_size) + meta.payload_size;
    const uint64_t footer_bytes = meta.checksummed ? kFooterBytes : 0;
    if (body_bytes > file_size - offset_u64 - meta.header_bytes ||
        footer_bytes > file_size - offset_u64 - meta.header_bytes - body_bytes) {
        return std::nullopt;
    }
    meta.record_bytes = meta.header_bytes + body_bytes + footer_bytes;
    if (meta.prev_offset < -1 ||
        (meta.prev_offset >= 0 && meta.prev_offset >= offset)) {
        return std::nullopt;
    }

    std::string key(meta.key_size, '\0');
    input.read(key.data(), static_cast<std::streamsize>(meta.key_size));
    if (!input.good()) {
        return std::nullopt;
    }
    if (out_key) *out_key = key;

    if (meta.checksummed) {
        std::vector<uint8_t> payload(static_cast<size_t>(meta.payload_size));
        if (!payload.empty()) {
            input.read(reinterpret_cast<char*>(payload.data()),
                       static_cast<std::streamsize>(payload.size()));
        }
        uint32_t commit_magic = 0;
        uint64_t committed_bytes = 0;
        if (!input || !read_le(input, commit_magic) ||
            !read_le(input, committed_bytes) ||
            commit_magic != kCommitMagic ||
            committed_bytes != meta.record_bytes ||
            record_checksum(key, payload, meta.prev_offset,
                            meta.timestamp_ms) != meta.checksum) {
            return std::nullopt;
        }
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
        static_cast<uint64_t>(offset) + meta.header_bytes + meta.key_size;
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
            bool recoverable_torn_tail = file_size - cursor < kHeaderBytes;
            input.clear();
            input.seekg(static_cast<std::streamoff>(cursor));
            uint32_t possible_magic = 0;
            if (!recoverable_torn_tail && read_le(input, possible_magic) &&
                possible_magic == kMagic) {
                uint32_t key_size = 0;
                uint64_t payload_size = 0;
                int64_t ignored_prev = -1;
                uint64_t ignored_timestamp = 0;
                uint64_t ignored_checksum = 0;
                if (!read_le(input, key_size) || !read_le(input, payload_size) ||
                    !read_le(input, ignored_prev) ||
                    !read_le(input, ignored_timestamp) ||
                    !read_le(input, ignored_checksum) || key_size == 0 ||
                    key_size > kMaxKeyBytes || payload_size > kMaxPayloadBytes) {
                    recoverable_torn_tail = true;
                } else {
                    const uint64_t record_bytes =
                        kHeaderBytes + static_cast<uint64_t>(key_size) +
                        payload_size + kFooterBytes;
                    if (record_bytes > file_size - cursor) {
                        recoverable_torn_tail = true;
                    } else {
                        input.clear();
                        input.seekg(static_cast<std::streamoff>(
                            cursor + record_bytes - kFooterBytes));
                        uint32_t commit_magic = 0;
                        uint64_t committed_bytes = 0;
                        if (!read_le(input, commit_magic) ||
                            !read_le(input, committed_bytes) ||
                            commit_magic != kCommitMagic ||
                            committed_bytes != record_bytes) {
                            recoverable_torn_tail = true;
                        }
                    }
                }
            }
            if (recoverable_torn_tail) {
                input.close();
                std::filesystem::resize_file(file_path_, cursor);
                durable_sync(file_path_);
                break;
            }
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
        cursor += meta->record_bytes;
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
