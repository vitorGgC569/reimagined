#include "../include/nsos/sha256.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nsos::integrity {
namespace {

class Sha256Accumulator {
public:
    void update(const unsigned char* data, std::size_t size) {
        if (size > 0 && data == nullptr) {
            throw std::invalid_argument(
                "SHA-256 input pointer is null for a non-empty buffer");
        }
        constexpr std::uint64_t kMaximumMessageBytes =
            std::numeric_limits<std::uint64_t>::max() / 8ull;
        if (size > kMaximumMessageBytes - total_size_) {
            throw std::overflow_error(
                "SHA-256 input exceeds the format's 64-bit length field");
        }
        total_size_ += static_cast<std::uint64_t>(size);
        std::size_t cursor = 0;
        while (cursor < size) {
            const std::size_t to_copy =
                std::min(size - cursor, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, data + cursor, to_copy);
            buffered_ += to_copy;
            cursor += to_copy;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::string final_hex() {
        const std::uint64_t total_bits = total_size_ * 8ull;
        block_[buffered_++] = 0x80u;
        if (buffered_ > 56) {
            while (buffered_ < block_.size()) {
                block_[buffered_++] = 0;
            }
            transform(block_.data());
            buffered_ = 0;
        }
        while (buffered_ < 56) {
            block_[buffered_++] = 0;
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            block_[buffered_++] = static_cast<unsigned char>(
                (total_bits >> shift) & 0xffu);
        }
        transform(block_.data());
        buffered_ = 0;

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::uint32_t value : state_) {
            out << std::setw(8) << value;
        }
        return out.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> kConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static std::uint32_t rotr(std::uint32_t value, std::uint32_t count) {
        return (value >> count) | (value << (32u - count));
    }
    static std::uint32_t choose(std::uint32_t x, std::uint32_t y,
                                std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }
    static std::uint32_t majority(std::uint32_t x, std::uint32_t y,
                                  std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static std::uint32_t big_sigma0(std::uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    static std::uint32_t big_sigma1(std::uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }
    static std::uint32_t small_sigma0(std::uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }
    static std::uint32_t small_sigma1(std::uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    void transform(const unsigned char* block) {
        std::uint32_t words[64] = {};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < 64; ++index) {
            words[index] =
                small_sigma1(words[index - 2]) + words[index - 7] +
                small_sigma0(words[index - 15]) + words[index - 16];
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < 64; ++index) {
            const std::uint32_t t1 =
                h + big_sigma1(e) + choose(e, f, g) +
                kConstants[index] + words[index];
            const std::uint32_t t2 = big_sigma0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<unsigned char, 64> block_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_size_ = 0;
};

}  // namespace

std::string sha256_hex(const void* data, std::size_t size) {
    Sha256Accumulator accumulator;
    accumulator.update(static_cast<const unsigned char*>(data), size);
    return accumulator.final_hex();
}

std::string sha256_hex_stream(
    const std::function<void(const Sha256Sink&)>& producer) {
    if (!producer) {
        throw std::invalid_argument(
            "SHA-256 stream producer is empty");
    }
    Sha256Accumulator accumulator;
    const Sha256Sink sink =
        [&](const void* data, std::size_t size) {
            accumulator.update(
                static_cast<const unsigned char*>(data), size);
        };
    producer(sink);
    return accumulator.final_hex();
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Could not hash file: " + path.string());
    }
    Sha256Accumulator accumulator;
    std::array<char, 4096> buffer{};
    while (input) {
        input.read(buffer.data(),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            accumulator.update(
                reinterpret_cast<const unsigned char*>(buffer.data()),
                static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Could not completely read file for SHA-256: " +
                                 path.string());
    }
    return accumulator.final_hex();
}

std::string sha256_file_prefix(const std::filesystem::path& path,
                               std::uint64_t size) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Could not hash file: " + path.string());
    }
    Sha256Accumulator accumulator;
    std::array<char, 4096> buffer{};
    std::uint64_t consumed = 0;
    while (consumed < size) {
        const std::uint64_t remaining = size - consumed;
        const std::streamsize wanted = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        input.read(buffer.data(), wanted);
        if (input.gcount() != wanted) {
            throw std::runtime_error(
                "File is shorter than the requested SHA-256 prefix: " +
                path.string());
        }
        accumulator.update(
            reinterpret_cast<const unsigned char*>(buffer.data()),
            static_cast<std::size_t>(wanted));
        consumed += static_cast<std::uint64_t>(wanted);
    }
    return accumulator.final_hex();
}

}  // namespace nsos::integrity
