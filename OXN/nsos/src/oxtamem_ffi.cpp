#include "../include/oxtamem_ffi.h"

#include <cstring>
#include <cstdlib>
#include <filesystem>
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

    const auto cwd = std::filesystem::current_path();
    candidates.push_back(cwd / library_name);
    candidates.push_back(cwd / "modules" / "oxtamem" / "oxta_engine" / "target" / "release" /
                         library_name);
    candidates.push_back(cwd / ".." / "modules" / "oxtamem" / "oxta_engine" / "target" /
                         "release" / library_name);
    candidates.push_back(cwd / ".." / ".." / "modules" / "oxtamem" / "oxta_engine" / "target" /
                         "release" / library_name);
    candidates.push_back(cwd / ".." / ".." / ".." / "modules" / "oxtamem" / "oxta_engine" /
                         "target" / "release" / library_name);
    return candidates;
}

LibraryHandle open_library(const std::filesystem::path& path) {
#ifdef _WIN32
    return LoadLibraryA(path.string().c_str());
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
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

} // namespace

struct OxtaMemFFI::Impl {
    using create_fn = void* (*)(const char*, uint64_t);
    using destroy_fn = void (*)(void*);
    using write_fn = bool (*)(void*, const char*, const uint8_t*, size_t);
    using read_latest_fn = bool (*)(void*, const char*, uint8_t**, size_t*);
    using recall_fn = bool (*)(void*, const char*, size_t, uint8_t**, size_t*);
    using free_buffer_fn = void (*)(uint8_t*, size_t);

    LibraryHandle library = nullptr;
    void* handle = nullptr;
    create_fn create = nullptr;
    destroy_fn destroy = nullptr;
    write_fn write = nullptr;
    read_latest_fn read_latest = nullptr;
    recall_fn recall = nullptr;
    free_buffer_fn free_buffer = nullptr;

    bool loaded() const {
        return library && create && destroy && write && read_latest && recall && free_buffer;
    }
};

OxtaMemFFI::OxtaMemFFI() : impl_(std::make_unique<Impl>()) {}

OxtaMemFFI::~OxtaMemFFI() {
    if (impl_->handle && impl_->destroy) {
        impl_->destroy(impl_->handle);
        impl_->handle = nullptr;
    }
    close_library(impl_->library);
}

bool OxtaMemFFI::load(const std::string& library_path) {
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

        impl_->library = library;
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
    }

    return false;
}

bool OxtaMemFFI::open(const std::string& store_path, uint64_t size_mb) {
    if (!load()) {
        return false;
    }
    if (impl_->handle && impl_->destroy) {
        impl_->destroy(impl_->handle);
        impl_->handle = nullptr;
    }
    impl_->handle = impl_->create(store_path.c_str(), size_mb);
    return impl_->handle != nullptr;
}

bool OxtaMemFFI::is_ready() const { return impl_->loaded() && impl_->handle != nullptr; }

bool OxtaMemFFI::write(const std::string& key, const std::vector<uint8_t>& value) {
    if (!is_ready()) {
        return false;
    }
    return impl_->write(impl_->handle, key.c_str(), value.data(), value.size());
}

std::vector<uint8_t> OxtaMemFFI::read_latest(const std::string& key) const {
    if (!is_ready()) {
        return {};
    }

    uint8_t* buffer = nullptr;
    size_t length = 0;
    if (!impl_->read_latest(impl_->handle, key.c_str(), &buffer, &length) || buffer == nullptr) {
        return {};
    }

    std::vector<uint8_t> result(buffer, buffer + length);
    impl_->free_buffer(buffer, length);
    return result;
}

std::vector<std::vector<uint8_t>> OxtaMemFFI::recall(const std::string& key, size_t depth) const {
    if (!is_ready()) {
        return {};
    }

    uint8_t* buffer = nullptr;
    size_t length = 0;
    if (!impl_->recall(impl_->handle, key.c_str(), depth, &buffer, &length) || buffer == nullptr ||
        length < sizeof(uint64_t)) {
        return {};
    }

    std::vector<std::vector<uint8_t>> values;
    const uint8_t* cursor = buffer;
    const uint8_t* end = buffer + length;
    const uint64_t count = read_u64_le(cursor);
    cursor += sizeof(uint64_t);

    for (uint64_t i = 0; i < count && cursor + sizeof(uint64_t) <= end; ++i) {
        const uint64_t item_size = read_u64_le(cursor);
        cursor += sizeof(uint64_t);
        if (cursor + item_size > end) {
            break;
        }
        values.emplace_back(cursor, cursor + item_size);
        cursor += item_size;
    }

    impl_->free_buffer(buffer, length);
    return values;
}

} // namespace nsos
