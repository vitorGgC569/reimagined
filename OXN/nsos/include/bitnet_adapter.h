#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace nsos {

class BitNetAdapter {
public:
  static void gemm_158bit_lut(const Tensor &input,
                              const std::vector<uint32_t> &packed_weights,
                              const std::vector<float> &act_scales,
                              float weight_scale,
                              Tensor &output);

  static void gemm_158bit_i8(const Tensor &input,
                             const std::vector<int8_t> &unpacked_weights,
                             const std::vector<int32_t> &weight_row_sums,
                             const std::vector<float> &act_scales,
                             float weight_scale,
                             Tensor &output);

  static void pack_weights_microsoft_style(const float *src,
                                           uint8_t *dst,
                                           int rows,
                                           int cols);

  static void unpack_weights_microsoft_style_to_i8(
      const std::vector<uint32_t> &packed_weights,
      int rows,
      int cols,
      std::vector<int8_t> &dst);
};

} // namespace nsos
