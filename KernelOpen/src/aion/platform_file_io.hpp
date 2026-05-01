#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <fcntl.h>

#if defined(AION_PLATFORM_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace Aion::platform_io {

inline int open_read_only(const std::string& filename) {
#if defined(AION_PLATFORM_WINDOWS)
    return _open(filename.c_str(), _O_RDONLY | _O_BINARY);
#else
    return open(filename.c_str(), O_RDONLY);
#endif
}

inline int close_file(int fd) {
#if defined(AION_PLATFORM_WINDOWS)
    return _close(fd);
#else
    return close(fd);
#endif
}

inline std::ptrdiff_t read_at_offset(int fd, void* buffer, std::size_t size, std::uint64_t offset) {
#if defined(AION_PLATFORM_WINDOWS)
    const auto seek_result = _lseeki64(fd, static_cast<__int64>(offset), SEEK_SET);
    if (seek_result < 0) {
        return -1;
    }
    return static_cast<std::ptrdiff_t>(_read(fd, buffer, static_cast<unsigned int>(size)));
#else
    return static_cast<std::ptrdiff_t>(pread(fd, buffer, size, static_cast<off_t>(offset)));
#endif
}

}  // namespace Aion::platform_io
