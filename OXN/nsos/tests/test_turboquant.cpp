#include "../include/turboquant.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  TqEngine* engine = nullptr;
  require(tq_engine_init(4, 7, &engine) == TQ_SUCCESS,
          "engine initialization failed");
  require(engine != nullptr, "engine is null");

  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  uint8_t* encoded = nullptr;
  size_t encoded_len = 0;
  require(tq_engine_encode(engine, values.data(), values.size(), &encoded,
                           &encoded_len) == TQ_SUCCESS,
          "encode failed");
  require(encoded != nullptr, "encoded pointer is null");

  float* decoded = nullptr;
  size_t decoded_len = 0;
  require(tq_engine_decode(engine, encoded, encoded_len, &decoded,
                           &decoded_len) == TQ_SUCCESS,
          "decode failed");
  require(decoded_len == values.size(), "decoded length mismatch");
  for (size_t i = 0; i < values.size(); ++i) {
    require(decoded[i] == values[i], "decoded value mismatch");
  }
  tq_free_f32(decoded, decoded_len);

  require(std::abs(tq_engine_dot(engine, values.data(), values.size(), encoded,
                                 encoded_len) -
                   30.0f) < 1e-6f,
          "dot mismatch");
  require(tq_engine_dot(engine, values.data(), values.size(), encoded,
                        encoded_len - 1) == 0.0f,
          "truncated dot payload was accepted");

  decoded = reinterpret_cast<float*>(1);
  decoded_len = 123;
  require(tq_engine_decode(engine, encoded, encoded_len - 1, &decoded,
                           &decoded_len) == TQ_ERR_INVALID_PAYLOAD,
          "truncated decode payload was accepted");
  require(decoded == nullptr && decoded_len == 0,
          "failed decode did not clear outputs");

  std::vector<uint8_t> hostile(encoded, encoded + encoded_len);
  const uint64_t hostile_len = (std::numeric_limits<uint64_t>::max)();
  std::memcpy(hostile.data() + sizeof(uint32_t) * 2, &hostile_len,
              sizeof(hostile_len));
  require(tq_engine_decode(engine, hostile.data(), hostile.size(), &decoded,
                           &decoded_len) == TQ_ERR_INVALID_HEADER,
          "hostile length header was accepted");
  require(tq_engine_dot(engine, values.data(), values.size(), hostile.data(),
                        hostile.size()) == 0.0f,
          "hostile dot header was accepted");

  uint8_t* bad_output = reinterpret_cast<uint8_t*>(1);
  size_t bad_output_len = 99;
  require(tq_engine_encode(engine, values.data(), values.size() - 1,
                           &bad_output,
                           &bad_output_len) == TQ_ERR_INVALID_DIMENSION,
          "wrong encode dimension was accepted");
  require(bad_output == nullptr && bad_output_len == 0,
          "failed encode did not clear outputs");

  tq_free_bytes(encoded, encoded_len);
  tq_engine_deinit(engine);
  std::cout << "TurboQuant fallback hardening test passed!\n";
  return 0;
}
