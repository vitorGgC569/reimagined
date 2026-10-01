#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace nsos::checkpoint_io {

inline uint64_t process_id() noexcept {
#ifdef _WIN32
    return static_cast<uint64_t>(GetCurrentProcessId());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

inline std::filesystem::path unique_temporary_path(
    const std::filesystem::path& destination) {
    static std::atomic<uint64_t> ordinal{0};
    const uint64_t sequence =
        ordinal.fetch_add(1, std::memory_order_relaxed);
    // Keep the basename independent of the destination filename and short.
    // The former destination/PID/timestamp/ordinal suffix crossed the
    // traditional Win32 MAX_PATH boundary in descriptive run directories even
    // when the final checkpoint path itself was valid. PID plus a process-local
    // monotonic ordinal is collision-free for concurrent writers. A stale file
    // from a crashed/reused PID is intentionally truncated by callers and is
    // never a committed checkpoint.
    return destination.parent_path() /
           (".tmp." + std::to_string(process_id()) + "." +
            std::to_string(sequence));
}

inline void flush_file(const std::filesystem::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "Could not open checkpoint for durable flush '" +
            path.string() + "' (Win32 error " +
            std::to_string(GetLastError()) + ")");
    }
    if (!FlushFileBuffers(handle)) {
        const DWORD error = GetLastError();
        CloseHandle(handle);
        throw std::runtime_error(
            "Could not durably flush checkpoint '" +
            path.string() + "' (Win32 error " +
            std::to_string(error) + ")");
    }
    if (!CloseHandle(handle)) {
        throw std::runtime_error(
            "Could not close checkpoint flush handle '" +
            path.string() + "' (Win32 error " +
            std::to_string(GetLastError()) + ")");
    }
#else
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Could not open checkpoint for durable flush '" +
            path.string() + "': " + std::strerror(errno));
    }
    if (fsync(descriptor) != 0) {
        const int error = errno;
        close(descriptor);
        throw std::runtime_error(
            "Could not durably flush checkpoint '" +
            path.string() + "': " + std::strerror(error));
    }
    if (close(descriptor) != 0) {
        throw std::runtime_error(
            "Could not close checkpoint flush descriptor '" +
            path.string() + "': " + std::strerror(errno));
    }
#endif
}

#ifndef _WIN32
inline void flush_parent_directory(
    const std::filesystem::path& destination) {
    const std::filesystem::path parent =
        destination.parent_path().empty()
            ? std::filesystem::current_path()
            : destination.parent_path();
    int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
    directory_flags |= O_DIRECTORY;
#endif
    const int descriptor = open(parent.c_str(), directory_flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Could not open checkpoint directory for durable flush '" +
            parent.string() + "': " + std::strerror(errno));
    }
    if (fsync(descriptor) != 0) {
        const int error = errno;
        close(descriptor);
        throw std::runtime_error(
            "Could not durably flush checkpoint directory '" +
            parent.string() + "': " + std::strerror(error));
    }
    if (close(descriptor) != 0) {
        throw std::runtime_error(
            "Could not close checkpoint directory descriptor '" +
            parent.string() + "': " + std::strerror(errno));
    }
}
#endif

inline void atomic_replace(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error(
            "Could not atomically replace checkpoint '" +
            destination.string() + "' (Win32 error " +
            std::to_string(error) + ")");
    }
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error(
            "Could not atomically replace checkpoint '" +
            destination.string() + "': " + error.message());
    }
    flush_parent_directory(destination);
#endif
}

}  // namespace nsos::checkpoint_io
