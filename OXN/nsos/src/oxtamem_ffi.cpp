#include "../include/oxtamem_ffi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#endif

namespace nsos {

namespace {

#ifdef _WIN32
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void*;
#endif

std::optional<std::string> read_env(const char* key) {
#ifdef _WIN32
    char* buffer = nullptr;
    size_t length = 0;
    if (_dupenv_s(&buffer, &length, key) != 0 || buffer == nullptr) {
        return std::nullopt;
    }
    std::string value(buffer);
    free(buffer);
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
#else
    const char* value = std::getenv(key);
    if (!value || !*value) {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::vector<std::filesystem::path> candidate_library_paths(const std::string& explicit_path) {
    std::vector<std::filesystem::path> candidates;
    if (!explicit_path.empty()) {
        candidates.emplace_back(explicit_path);
    }

    if (const auto env_path = read_env("NSOS_OXTAMEM_LIBRARY")) {
        candidates.emplace_back(*env_path);
    }

#ifdef NSOS_OXTAMEM_DEFAULT_LIBRARY
    candidates.emplace_back(NSOS_OXTAMEM_DEFAULT_LIBRARY);
#endif

#ifdef _WIN32
    const char* library_name = "oxta_mem.dll";
#elif __APPLE__
    const char* library_name = "liboxta_mem.dylib";
#else
    const char* library_name = "liboxta_mem.so";
#endif

    // SECURITY: do NOT search the current working directory or relative paths
    // by default — loading a native library from CWD/relative dirs is a classic
    // DLL/.so planting vector (an attacker who can drop oxta_mem.dll next to the
    // process gets code execution).  Production resolves the library via the
    // explicit argument, the NSOS_OXTAMEM_LIBRARY env, or the compile-time
    // absolute NSOS_OXTAMEM_DEFAULT_LIBRARY above.  The legacy CWD/relative
    // search is opt-in for local development only.
    if (const auto allow_cwd = read_env("NSOS_OXTAMEM_ALLOW_CWD");
        allow_cwd && (*allow_cwd == "1" || *allow_cwd == "true")) {
        const auto cwd = std::filesystem::current_path();
        candidates.push_back(cwd / library_name);
        candidates.push_back(cwd / "modules" / "oxtamem" / "oxta_engine" / "target" /
                             "release" / library_name);
        candidates.push_back(cwd / ".." / "modules" / "oxtamem" / "oxta_engine" / "target" /
                             "release" / library_name);
        candidates.push_back(cwd / ".." / ".." / "modules" / "oxtamem" / "oxta_engine" /
                             "target" / "release" / library_name);
        candidates.push_back(cwd / ".." / ".." / ".." / "modules" / "oxtamem" / "oxta_engine" /
                             "target" / "release" / library_name);
    }
    return candidates;
}

LibraryHandle open_library(const std::filesystem::path& path) {
#ifdef _WIN32
    return LoadLibraryExW(path.c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                              LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    return dlopen(path.string().c_str(), RTLD_NOW);
#endif
}

void close_library(LibraryHandle handle) {
    if (!handle) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

void* resolve_symbol(LibraryHandle handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(handle, name));
#else
    return dlsym(handle, name);
#endif
}

uint64_t read_u64_le(const uint8_t* ptr) {
    uint64_t value = 0;
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        value |= static_cast<uint64_t>(ptr[byte]) << (byte * 8);
    }
    return value;
}

bool valid_c_string_input(const std::string& value, size_t max_bytes,
                          bool allow_empty = false) {
    return (allow_empty || !value.empty()) && value.size() <= max_bytes &&
           value.find('\0') == std::string::npos;
}

} // namespace

struct OxtaMemFFI::Impl {
    using abi_version_fn = uint32_t (*)();
    using last_error_fn = const char* (*)();
    using create_fn = void* (*)(const char*, uint64_t);
    using destroy_fn = void (*)(void*);
    using write_fn = bool (*)(void*, const char*, const uint8_t*, size_t);
    using write_with_vector_fn =
        bool (*)(void*, const char*, const uint8_t*, size_t,
                 const float*, size_t);
    using read_latest_fn = bool (*)(void*, const char*, uint8_t**, size_t*);
    using recall_fn = bool (*)(void*, const char*, size_t, uint8_t**, size_t*);
    using search_similar_fn =
        bool (*)(void*, const float*, size_t, size_t, uint8_t**, size_t*);
    using free_buffer_fn = void (*)(uint8_t*, size_t);

    static constexpr uint32_t kExpectedAbiVersion = 2;

    mutable std::mutex mutex;
    mutable std::string diagnostic;
    LibraryHandle library = nullptr;
    void* handle = nullptr;
    abi_version_fn abi_version = nullptr;
    last_error_fn backend_last_error = nullptr;
    create_fn create = nullptr;
    destroy_fn destroy = nullptr;
    write_fn write = nullptr;
    write_with_vector_fn write_with_vector = nullptr;
    read_latest_fn read_latest = nullptr;
    recall_fn recall = nullptr;
    search_similar_fn search_similar = nullptr;
    free_buffer_fn free_buffer = nullptr;

    bool loaded() const {
        return library && abi_version && backend_last_error && create && destroy &&
               write && write_with_vector && read_latest && recall &&
               search_similar && free_buffer &&
               abi_version() == kExpectedAbiVersion;
    }

    std::string capture_backend_error(const std::string& fallback) const {
        if (backend_last_error) {
            const char* message = backend_last_error();
            if (message && *message) {
                diagnostic = message;
                return diagnostic;
            }
        }
        diagnostic = fallback;
        return diagnostic;
    }

    void clear_symbols() {
        abi_version = nullptr;
        backend_last_error = nullptr;
        create = nullptr;
        destroy = nullptr;
        write = nullptr;
        write_with_vector = nullptr;
        read_latest = nullptr;
        recall = nullptr;
        search_similar = nullptr;
        free_buffer = nullptr;
    }
};

OxtaMemFFI::OxtaMemFFI() : impl_(std::make_unique<Impl>()) {}

OxtaMemFFI::~OxtaMemFFI() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->handle && impl_->destroy) {
        impl_->destroy(impl_->handle);
        impl_->handle = nullptr;
    }
    close_library(impl_->library);
}

bool OxtaMemFFI::load(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->loaded()) {
        impl_->diagnostic.clear();
        return true;
    }

    const auto candidates = candidate_library_paths(library_path);
    if (candidates.empty()) {
        impl_->diagnostic =
            "No OxtaMem library path is configured. Set an explicit path or "
            "NSOS_OXTAMEM_LIBRARY.";
        return false;
    }

    for (const auto& candidate : candidates) {
        if (!std::filesystem::exists(candidate)) {
            impl_->diagnostic =
                "OxtaMem library does not exist: " + candidate.string();
            continue;
        }

        LibraryHandle library = open_library(candidate);
        if (!library) {
            impl_->diagnostic =
                "Failed to load OxtaMem library: " + candidate.string();
            continue;
        }

        impl_->clear_symbols();
        impl_->library = library;
        impl_->abi_version = reinterpret_cast<Impl::abi_version_fn>(
            resolve_symbol(library, "oxtamem_abi_version"));
        impl_->backend_last_error = reinterpret_cast<Impl::last_error_fn>(
            resolve_symbol(library, "oxtamem_last_error"));
        impl_->create = reinterpret_cast<Impl::create_fn>(resolve_symbol(library, "oxtamem_create"));
        impl_->destroy =
            reinterpret_cast<Impl::destroy_fn>(resolve_symbol(library, "oxtamem_destroy"));
        impl_->write = reinterpret_cast<Impl::write_fn>(resolve_symbol(library, "oxtamem_write"));
        impl_->write_with_vector =
            reinterpret_cast<Impl::write_with_vector_fn>(
                resolve_symbol(library, "oxtamem_write_with_vector"));
        impl_->read_latest = reinterpret_cast<Impl::read_latest_fn>(
            resolve_symbol(library, "oxtamem_read_latest"));
        impl_->recall =
            reinterpret_cast<Impl::recall_fn>(resolve_symbol(library, "oxtamem_recall"));
        impl_->search_similar =
            reinterpret_cast<Impl::search_similar_fn>(
                resolve_symbol(library, "oxtamem_search_similar"));
        impl_->free_buffer = reinterpret_cast<Impl::free_buffer_fn>(
            resolve_symbol(library, "oxtamem_free_buffer"));

        if (impl_->loaded()) {
            impl_->diagnostic.clear();
            return true;
        }

        if (impl_->abi_version &&
            impl_->abi_version() != Impl::kExpectedAbiVersion) {
            impl_->diagnostic =
                "OxtaMem ABI mismatch: expected " +
                std::to_string(Impl::kExpectedAbiVersion) + ", got " +
                std::to_string(impl_->abi_version());
        } else {
            impl_->diagnostic =
                "OxtaMem library is missing one or more required ABI v2 symbols: " +
                candidate.string();
        }
        close_library(library);
        impl_->library = nullptr;
        impl_->clear_symbols();
    }

    return false;
}

bool OxtaMemFFI::open(const std::string& store_path, uint64_t size_mb) {
    if (!valid_c_string_input(store_path, 32768) || size_mb < 1 || size_mb > 32768) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->diagnostic =
            "Invalid OxtaMem store path or size_mb (expected 1..32768)";
        return false;
    }
    if (!load()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->handle && impl_->destroy) {
        impl_->destroy(impl_->handle);
        impl_->handle = nullptr;
    }
    impl_->handle = impl_->create(store_path.c_str(), size_mb);
    if (!impl_->handle) {
        impl_->capture_backend_error("OxtaMem store open failed");
        return false;
    }
    impl_->diagnostic.clear();
    return true;
}

bool OxtaMemFFI::is_ready() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->loaded() && impl_->handle != nullptr;
}

std::string OxtaMemFFI::last_error() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->diagnostic;
}

bool OxtaMemFFI::write(const std::string& key, const std::vector<uint8_t>& value) {
    if (!valid_c_string_input(key, 1024) || value.size() > 16ull * 1024ull * 1024ull) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->diagnostic = "Invalid OxtaMem key or value length";
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) {
        impl_->diagnostic = "OxtaMem store is not open";
        return false;
    }
    const uint8_t* data = value.empty() ? nullptr : value.data();
    if (!impl_->write(impl_->handle, key.c_str(), data, value.size())) {
        impl_->capture_backend_error("OxtaMem write failed");
        return false;
    }
    impl_->diagnostic.clear();
    return true;
}

bool OxtaMemFFI::write_with_vector(
    const std::string& key, const std::vector<uint8_t>& value,
    const std::vector<float>& vector) {
    if (!valid_c_string_input(key, 1024) ||
        value.size() > 16ull * 1024ull * 1024ull || vector.empty() ||
        vector.size() > 1'048'576 ||
        !std::all_of(vector.begin(), vector.end(),
                     [](float item) { return std::isfinite(item); })) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->diagnostic =
            "Invalid OxtaMem key, value, or similarity vector";
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) {
        impl_->diagnostic = "OxtaMem store is not open";
        return false;
    }
    const uint8_t* data = value.empty() ? nullptr : value.data();
    if (!impl_->write_with_vector(
            impl_->handle, key.c_str(), data, value.size(), vector.data(),
            vector.size())) {
        impl_->capture_backend_error("OxtaMem vector write failed");
        return false;
    }
    impl_->diagnostic.clear();
    return true;
}

std::vector<uint8_t> OxtaMemFFI::read_latest(const std::string& key) const {
    if (!valid_c_string_input(key, 1024)) {
        throw std::invalid_argument("Invalid OxtaMem key");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) {
        throw std::runtime_error("OxtaMem store is not open");
    }

    uint8_t* buffer = nullptr;
    size_t length = 0;
    const bool ok = impl_->read_latest(impl_->handle, key.c_str(), &buffer, &length);
    if (!ok) {
        if (buffer) impl_->free_buffer(buffer, length);
        const std::string error =
            impl_->capture_backend_error("OxtaMem read_latest failed");
        if (error == "not found") {
            return {};
        }
        throw std::runtime_error(error);
    }
    if ((buffer == nullptr && length != 0) ||
        length > 16ull * 1024ull * 1024ull) {
        if (buffer) impl_->free_buffer(buffer, length);
        impl_->diagnostic = "Malformed OxtaMem read_latest buffer";
        throw std::runtime_error(impl_->diagnostic);
    }

    std::vector<uint8_t> result;
    try {
        if (length > 0) result.assign(buffer, buffer + length);
    } catch (...) {
        if (buffer) impl_->free_buffer(buffer, length);
        throw;
    }
    if (buffer) impl_->free_buffer(buffer, length);
    impl_->diagnostic.clear();
    return result;
}

std::vector<std::vector<uint8_t>> OxtaMemFFI::recall(const std::string& key, size_t depth) const {
    if (!valid_c_string_input(key, 1024) || depth > 1024) {
        throw std::invalid_argument("Invalid OxtaMem key or recall depth");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) {
        throw std::runtime_error("OxtaMem store is not open");
    }

    uint8_t* buffer = nullptr;
    size_t length = 0;
    const bool ok = impl_->recall(impl_->handle, key.c_str(), depth, &buffer, &length);
    if (!ok) {
        if (buffer) impl_->free_buffer(buffer, length);
        throw std::runtime_error(
            impl_->capture_backend_error("OxtaMem recall failed"));
    }
    if (buffer == nullptr || length < sizeof(uint64_t) ||
        length > 64ull * 1024ull * 1024ull) {
        if (buffer) impl_->free_buffer(buffer, length);
        impl_->diagnostic = "Malformed OxtaMem recall buffer";
        throw std::runtime_error(impl_->diagnostic);
    }

    std::vector<std::vector<uint8_t>> values;
    bool valid = true;
    try {
        const uint8_t* cursor = buffer;
        const uint8_t* end = buffer + length;
        const uint64_t count = read_u64_le(cursor);
        cursor += sizeof(uint64_t);
        if (count > depth || count > 1024) valid = false;
        values.reserve(valid ? static_cast<size_t>(count) : 0);
        for (uint64_t i = 0; valid && i < count; ++i) {
            if (static_cast<size_t>(end - cursor) < sizeof(uint64_t)) {
                valid = false;
                break;
            }
            const uint64_t item_size = read_u64_le(cursor);
            cursor += sizeof(uint64_t);
            if (item_size > 16ull * 1024ull * 1024ull ||
                item_size > static_cast<uint64_t>(end - cursor)) {
                valid = false;
                break;
            }
            const size_t item_bytes = static_cast<size_t>(item_size);
            values.emplace_back(cursor, cursor + item_bytes);
            cursor += item_bytes;
        }
        valid = valid && cursor == end;
    } catch (...) {
        impl_->free_buffer(buffer, length);
        throw;
    }

    impl_->free_buffer(buffer, length);
    if (!valid) {
        impl_->diagnostic = "Malformed OxtaMem recall payload";
        throw std::runtime_error(impl_->diagnostic);
    }
    impl_->diagnostic.clear();
    return values;
}

std::vector<std::vector<uint8_t>> OxtaMemFFI::search_similar(
    const std::vector<float>& vector, size_t top_k) const {
    if (vector.empty() || vector.size() > 1'048'576 || top_k > 4096 ||
        !std::all_of(vector.begin(), vector.end(),
                     [](float item) { return std::isfinite(item); })) {
        throw std::invalid_argument(
            "Invalid OxtaMem similarity vector or top_k");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) {
        throw std::runtime_error("OxtaMem store is not open");
    }

    uint8_t* buffer = nullptr;
    size_t length = 0;
    const bool ok = impl_->search_similar(
        impl_->handle, vector.data(), vector.size(), top_k, &buffer, &length);
    if (!ok) {
        if (buffer) impl_->free_buffer(buffer, length);
        throw std::runtime_error(
            impl_->capture_backend_error("OxtaMem similarity search failed"));
    }
    if (buffer == nullptr || length < sizeof(uint64_t) ||
        length > 64ull * 1024ull * 1024ull) {
        if (buffer) impl_->free_buffer(buffer, length);
        impl_->diagnostic = "Malformed OxtaMem similarity-search buffer";
        throw std::runtime_error(impl_->diagnostic);
    }

    std::vector<std::vector<uint8_t>> values;
    bool valid = true;
    try {
        const uint8_t* cursor = buffer;
        const uint8_t* end = buffer + length;
        const uint64_t count = read_u64_le(cursor);
        cursor += sizeof(uint64_t);
        if (count > top_k || count > 4096) valid = false;
        values.reserve(valid ? static_cast<size_t>(count) : 0);
        for (uint64_t i = 0; valid && i < count; ++i) {
            if (static_cast<size_t>(end - cursor) < sizeof(uint64_t)) {
                valid = false;
                break;
            }
            const uint64_t item_size = read_u64_le(cursor);
            cursor += sizeof(uint64_t);
            if (item_size > 16ull * 1024ull * 1024ull ||
                item_size > static_cast<uint64_t>(end - cursor)) {
                valid = false;
                break;
            }
            const size_t item_bytes = static_cast<size_t>(item_size);
            values.emplace_back(cursor, cursor + item_bytes);
            cursor += item_bytes;
        }
        valid = valid && cursor == end;
    } catch (...) {
        impl_->free_buffer(buffer, length);
        throw;
    }
    impl_->free_buffer(buffer, length);
    if (!valid) {
        impl_->diagnostic = "Malformed OxtaMem similarity-search payload";
        throw std::runtime_error(impl_->diagnostic);
    }
    impl_->diagnostic.clear();
    return values;
}

} // namespace nsos
