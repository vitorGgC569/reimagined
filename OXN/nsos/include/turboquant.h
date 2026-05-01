#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque Engine pointer from Zig
typedef struct TqEngine TqEngine;

#define TQ_SUCCESS 0
#define TQ_ERR_INVALID_DIMENSION -1
#define TQ_ERR_OUT_OF_MEMORY -2
#define TQ_ERR_INVALID_HEADER -3
#define TQ_ERR_INVALID_PAYLOAD -4

// Interface bindings exported by turboquant c_api.zig
int32_t tq_engine_init(size_t dim, uint32_t seed, TqEngine** out_engine);
void tq_engine_deinit(TqEngine* engine);

int32_t tq_engine_encode(TqEngine* engine, const float* x_ptr, size_t x_len, uint8_t** out_bytes, size_t* out_len);
int32_t tq_engine_decode(TqEngine* engine, const uint8_t* compressed_ptr, size_t compressed_len, float** out_f32, size_t* out_len);
float tq_engine_dot(TqEngine* engine, const float* q_ptr, size_t q_len, const uint8_t* compressed_ptr, size_t compressed_len);

void tq_free_bytes(uint8_t* ptr, size_t len);
void tq_free_f32(float* ptr, size_t len);

#ifdef __cplusplus
} // extern "C"

// Modern C++ RAII Wrapper
#include <vector>
#include <stdexcept>
#include <memory>

namespace nsos {
namespace tq {

class TurboQuantEngine {
public:
    TurboQuantEngine(size_t dim, uint32_t seed = 12345) {
        if (tq_engine_init(dim, seed, &engine_) != TQ_SUCCESS) {
            throw std::runtime_error("Failed to initialize TurboQuant Engine. Ensure dim is even and > 0.");
        }
    }

    ~TurboQuantEngine() {
        if (engine_) {
            tq_engine_deinit(engine_);
            engine_ = nullptr;
        }
    }

    std::vector<uint8_t> encode(const std::vector<float>& input) {
        uint8_t* out_bytes = nullptr;
        size_t out_len = 0;
        int32_t res = tq_engine_encode(engine_, input.data(), input.size(), &out_bytes, &out_len);
        if (res != TQ_SUCCESS) {
            throw std::runtime_error("TurboQuant encoding failed");
        }
        
        std::vector<uint8_t> result(out_bytes, out_bytes + out_len);
        tq_free_bytes(out_bytes, out_len); // Free allocation from Zig
        return result;
    }

    std::vector<float> decode(const std::vector<uint8_t>& compressed) {
        float* out_f32 = nullptr;
        size_t out_len = 0;
        int32_t res = tq_engine_decode(engine_, compressed.data(), compressed.size(), &out_f32, &out_len);
        if (res != TQ_SUCCESS) {
            throw std::runtime_error("TurboQuant decoding failed");
        }

        std::vector<float> result(out_f32, out_f32 + out_len);
        tq_free_f32(out_f32, out_len); // Free allocation from Zig
        return result;
    }

    float dot(const std::vector<float>& query, const std::vector<uint8_t>& compressed) {
        return tq_engine_dot(engine_, query.data(), query.size(), compressed.data(), compressed.size());
    }

private:
    TqEngine* engine_ = nullptr;
};

} // namespace tq
} // namespace nsos
#endif

#endif // TURBOQUANT_H
