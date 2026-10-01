#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace nsos::integrity {

// Lowercase SHA-256 (FIPS 180-4) of an exact byte sequence.
std::string sha256_hex(const void* data, std::size_t size);

using Sha256Sink =
    std::function<void(const void*, std::size_t)>;

// Incremental producer interface for canonical hashes that must not
// materialize a second full copy of a large tensor or artifact.
std::string sha256_hex_stream(
    const std::function<void(const Sha256Sink&)>& producer);

// Streaming SHA-256 of a file. Throws when the file cannot be read.
std::string sha256_file(const std::filesystem::path& path);

// Streaming SHA-256 of exactly the first `size` file bytes. Throws when the
// file is shorter or cannot be read.
std::string sha256_file_prefix(const std::filesystem::path& path,
                               std::uint64_t size);

}  // namespace nsos::integrity
