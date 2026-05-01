#include "../include/turboquant.h"

#ifndef NSOS_USE_EXTERNAL_TURBOQUANT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

struct TqEngine {
    size_t dim;
    uint32_t seed;
};

namespace {

constexpr uint32_t kMagic = 0x54514642u; // TQFB
constexpr uint32_t kVersion = 1u;

struct Header {
    uint32_t magic;
    uint32_t version;
    uint64_t len;
};

bool valid_engine(const TqEngine* engine) {
    return engine != nullptr && engine->dim > 0 && (engine->dim % 2u) == 0u;
}

} // namespace

extern "C" {

int32_t tq_engine_init(size_t dim, uint32_t seed, TqEngine** out_engine) {
    if (!out_engine || dim == 0 || (dim % 2u) != 0u) {
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
    if (!valid_engine(engine) || !x_ptr || !out_bytes || !out_len || x_len == 0) {
        return TQ_ERR_INVALID_DIMENSION;
    }

    const size_t payload_bytes = x_len * sizeof(float);
    const size_t total_bytes = sizeof(Header) + payload_bytes;
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
    if (!valid_engine(engine) || !compressed_ptr || !out_f32 || !out_len ||
        compressed_len < sizeof(Header)) {
        return TQ_ERR_INVALID_HEADER;
    }

    Header header{};
    std::memcpy(&header, compressed_ptr, sizeof(Header));
    if (header.magic != kMagic || header.version != kVersion) {
        return TQ_ERR_INVALID_HEADER;
    }

    const size_t expected_bytes = sizeof(Header) + static_cast<size_t>(header.len) * sizeof(float);
    if (compressed_len != expected_bytes) {
        return TQ_ERR_INVALID_PAYLOAD;
    }

    float* output = new (std::nothrow) float[static_cast<size_t>(header.len)];
    if (!output) {
        return TQ_ERR_OUT_OF_MEMORY;
    }

    std::memcpy(output, compressed_ptr + sizeof(Header),
                static_cast<size_t>(header.len) * sizeof(float));
    *out_f32 = output;
    *out_len = static_cast<size_t>(header.len);
    return TQ_SUCCESS;
}

float tq_engine_dot(TqEngine* engine, const float* q_ptr, size_t q_len,
                    const uint8_t* compressed_ptr, size_t compressed_len) {
    if (!valid_engine(engine) || !q_ptr || !compressed_ptr || compressed_len < sizeof(Header)) {
        return 0.0f;
    }

    Header header{};
    std::memcpy(&header, compressed_ptr, sizeof(Header));
    if (header.magic != kMagic || header.version != kVersion) {
        return 0.0f;
    }

    const size_t payload_len = std::min(q_len, static_cast<size_t>(header.len));
    const float* decoded =
        reinterpret_cast<const float*>(compressed_ptr + sizeof(Header));

    float dot = 0.0f;
    for (size_t i = 0; i < payload_len; ++i) {
        dot += q_ptr[i] * decoded[i];
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
