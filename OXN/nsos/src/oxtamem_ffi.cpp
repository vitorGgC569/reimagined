#include "../include/oxtamem_ffi.h"

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
    using create_fn = void* (*)(const char*, uint64_t);
    using destroy_fn = void (*)(void*);
    using write_fn = bool (*)(void*, const char*, const uint8_t*, size_t);
    using read_latest_fn = bool (*)(void*, const char*, uint8_t**, size_t*);
    using recall_fn = bool (*)(void*, const char*, size_t, uint8_t**, size_t*);
    using free_buffer_fn = void (*)(uint8_t*, size_t);

    static constexpr uint32_t kExpectedAbiVersion = 1;

    mutable std::mutex mutex;
    LibraryHandle library = nullptr;
    void* handle = nullptr;
    abi_version_fn abi_version = nullptr;
    create_fn create = nullptr;
    destroy_fn destroy = nullptr;
    write_fn write = nullptr;
    read_latest_fn read_latest = nullptr;
    recall_fn recall = nullptr;
    free_buffer_fn free_buffer = nullptr;

    bool loaded() const {
        return library && abi_version && create && destroy && write && read_latest && recall &&
               free_buffer && abi_version() == kExpectedAbiVersion;
    }

    void clear_symbols() {
        abi_version = nullptr;
        create = nullptr;
        destroy = nullptr;
        write = nullptr;
        read_latest = nullptr;
        recall = nullptr;
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
        return true;
    }

    for (const auto& candidate : candidate_library_paths(library_path)) {
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        LibraryHandle library = open_library(candidate);
        if (!library) {
            continue;
        }

        impl_->clear_symbols();
        impl_->library = library;
        impl_->abi_version = reinterpret_cast<Impl::abi_version_fn>(
            resolve_symbol(library, "oxtamem_abi_version"));
        impl_->create = reinterpret_cast<Impl::create_fn>(resolve_symbol(library, "oxtamem_create"));
        impl_->destroy =
            reinterpret_cast<Impl::destroy_fn>(resolve_symbol(library, "oxtamem_destroy"));
        impl_->write = reinterpret_cast<Impl::write_fn>(resolve_symbol(library, "oxtamem_write"));
        impl_->read_latest = reinterpret_cast<Impl::read_latest_fn>(
            resolve_symbol(library, "oxtamem_read_latest"));
        impl_->recall =
            reinterpret_cast<Impl::recall_fn>(resolve_symbol(library, "oxtamem_recall"));
        impl_->free_buffer = reinterpret_cast<Impl::free_buffer_fn>(
            resolve_symbol(library, "oxtamem_free_buffer"));

        if (impl_->loaded()) {
            return true;
        }

        close_library(library);
        impl_->library = nullptr;
        impl_->clear_symbols();
    }

    return false;
}

bool OxtaMemFFI::open(const std::string& store_path, uint64_t size_mb) {
    if (!valid_c_string_input(store_path, 32768) || size_mb < 1 || size_mb > 32768) {
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
    return impl_->handle != nullptr;
}

bool OxtaMemFFI::is_ready() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->loaded() && impl_->handle != nullptr;
}

bool OxtaMemFFI::write(const std::string& key, const std::vector<uint8_t>& value) {
    if (!valid_c_string_input(key, 1024) || value.size() > 16ull * 1024ull * 1024ull) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) return false;
    const uint8_t* data = value.empty() ? nullptr : value.data();
    return impl_->write(impl_->handle, key.c_str(), data, value.size());
}

std::vector<uint8_t> OxtaMemFFI::read_latest(const std::string& key) const {
    if (!valid_c_string_input(key, 1024)) {
        return {};
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) return {};

    uint8_t* buffer = nullptr;
    size_t length = 0;
    const bool ok = impl_->read_latest(impl_->handle, key.c_str(), &buffer, &length);
    if (!ok || (buffer == nullptr && length != 0) || length > 16ull * 1024ull * 1024ull) {
        if (buffer) impl_->free_buffer(buffer, length);
        return {};
    }

    std::vector<uint8_t> result;
    try {
        if (length > 0) result.assign(buffer, buffer + length);
    } catch (...) {
        if (buffer) impl_->free_buffer(buffer, length);
        throw;
    }
    if (buffer) impl_->free_buffer(buffer, length);
    return result;
}

std::vector<std::vector<uint8_t>> OxtaMemFFI::recall(const std::string& key, size_t depth) const {
    if (!valid_c_string_input(key, 1024) || depth > 1024) {
        return {};
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->loaded() || !impl_->handle) return {};

    uint8_t* buffer = nullptr;
    size_t length = 0;
    const bool ok = impl_->recall(impl_->handle, key.c_str(), depth, &buffer, &length);
    if (!ok || buffer == nullptr || length < sizeof(uint64_t) ||
        length > 64ull * 1024ull * 1024ull) {
        if (buffer) impl_->free_buffer(buffer, length);
        return {};
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
    return valid ? values : std::vector<std::vector<uint8_t>>{};
}

} // namespace nsos
