#include "../include/turboquant.h"

#ifndef NSOS_USE_EXTERNAL_TURBOQUANT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

struct TqEngine {
    size_t dim;
    uint32_t seed;
};

namespace {

constexpr uint32_t kMagic = 0x54514642u; // TQFB
constexpr uint32_t kVersion = 1u;
constexpr size_t kMaxDimension = 16u * 1024u * 1024u;

struct Header {
    uint32_t magic;
    uint32_t version;
    uint64_t len;
};

bool valid_engine(const TqEngine* engine) {
    return engine != nullptr && engine->dim > 0 &&
           engine->dim <= kMaxDimension && (engine->dim % 2u) == 0u;
}

bool encoded_size(size_t elements, size_t& payload_bytes,
                  size_t& total_bytes) {
    if (elements > kMaxDimension ||
        elements > (std::numeric_limits<size_t>::max)() / sizeof(float)) {
        return false;
    }
    payload_bytes = elements * sizeof(float);
    if (payload_bytes >
        (std::numeric_limits<size_t>::max)() - sizeof(Header)) {
        return false;
    }
    total_bytes = sizeof(Header) + payload_bytes;
    return true;
}

} // namespace

extern "C" {

int32_t tq_engine_init(size_t dim, uint32_t seed, TqEngine** out_engine) {
    if (!out_engine) {
        return TQ_ERR_INVALID_DIMENSION;
    }
    *out_engine = nullptr;
    if (dim == 0 || dim > kMaxDimension || (dim % 2u) != 0u) {
        return TQ_ERR_INVALID_DIMENSION;
    }

    TqEngine* engine = new (std::nothrow) TqEngine{dim, seed};
    if (!engine) {
        return TQ_ERR_OUT_OF_MEMORY;
    }

    *out_engine = engine;
    return TQ_SUCCESS;
}

void tq_engine_deinit(TqEngine* engine) { delete engine; }

int32_t tq_engine_encode(TqEngine* engine, const float* x_ptr, size_t x_len,
                         uint8_t** out_bytes, size_t* out_len) {
    if (!out_bytes || !out_len) {
        return TQ_ERR_INVALID_DIMENSION;
    }
    *out_bytes = nullptr;
    *out_len = 0;
    if (!valid_engine(engine) || !x_ptr || x_len != engine->dim) {
        return TQ_ERR_INVALID_DIMENSION;
    }

    size_t payload_bytes = 0;
    size_t total_bytes = 0;
    if (!encoded_size(x_len, payload_bytes, total_bytes)) {
        return TQ_ERR_INVALID_DIMENSION;
    }
    uint8_t* bytes = new (std::nothrow) uint8_t[total_bytes];
    if (!bytes) {
        return TQ_ERR_OUT_OF_MEMORY;
    }

    Header header{kMagic, kVersion, static_cast<uint64_t>(x_len)};
    std::memcpy(bytes, &header, sizeof(Header));
    std::memcpy(bytes + sizeof(Header), x_ptr, payload_bytes);

    *out_bytes = bytes;
    *out_len = total_bytes;
    return TQ_SUCCESS;
}

int32_t tq_engine_decode(TqEngine* engine, const uint8_t* compressed_ptr,
                         size_t compressed_len, float** out_f32, size_t* out_len) {
    if (!out_f32 || !out_len) {
        return TQ_ERR_INVALID_HEADER;
    }
    *out_f32 = nullptr;
    *out_len = 0;
    if (!valid_engine(engine) || !compressed_ptr ||
        compressed_len < sizeof(Header)) {
        return TQ_ERR_INVALID_HEADER;
    }

    Header header{};
    std::memcpy(&header, compressed_ptr, sizeof(Header));
    if (header.magic != kMagic || header.version != kVersion ||
        header.len != static_cast<uint64_t>(engine->dim)) {
        return TQ_ERR_INVALID_HEADER;
    }

    size_t payload_bytes = 0;
    size_t expected_bytes = 0;
    if (!encoded_size(engine->dim, payload_bytes, expected_bytes)) {
        return TQ_ERR_INVALID_HEADER;
    }
    if (compressed_len != expected_bytes) {
        return TQ_ERR_INVALID_PAYLOAD;
    }

    float* output = new (std::nothrow) float[engine->dim];
    if (!output) {
        return TQ_ERR_OUT_OF_MEMORY;
    }

    std::memcpy(output, compressed_ptr + sizeof(Header),
                payload_bytes);
    *out_f32 = output;
    *out_len = engine->dim;
    return TQ_SUCCESS;
}

float tq_engine_dot(TqEngine* engine, const float* q_ptr, size_t q_len,
                    const uint8_t* compressed_ptr, size_t compressed_len) {
    if (!valid_engine(engine) || !q_ptr || q_len != engine->dim ||
        !compressed_ptr || compressed_len < sizeof(Header)) {
        return 0.0f;
    }

    Header header{};
    std::memcpy(&header, compressed_ptr, sizeof(Header));
    if (header.magic != kMagic || header.version != kVersion ||
        header.len != static_cast<uint64_t>(engine->dim)) {
        return 0.0f;
    }

    size_t payload_bytes = 0;
    size_t expected_bytes = 0;
    if (!encoded_size(engine->dim, payload_bytes, expected_bytes) ||
        compressed_len != expected_bytes) {
        return 0.0f;
    }

    float dot = 0.0f;
    for (size_t i = 0; i < engine->dim; ++i) {
        float decoded = 0.0f;
        std::memcpy(&decoded,
                    compressed_ptr + sizeof(Header) + i * sizeof(float),
                    sizeof(float));
        dot += q_ptr[i] * decoded;
    }
    return dot;
}

void tq_free_bytes(uint8_t* ptr, size_t len) {
    (void)len;
    delete[] ptr;
}

void tq_free_f32(float* ptr, size_t len) {
    (void)len;
    delete[] ptr;
}

} // extern "C"

#endif
