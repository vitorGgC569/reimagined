#pragma once
#include "tensor.h"
#include <vector>

namespace nsos {
// No gradient filtering. Losses are weighted sums; the caller owns the
// denominator and loss scaling. Negative IDs reproduce the optional RUL loss.
struct TiledCrossEntropyOptions {
    std::vector<int> targets;
    std::vector<float> weights;
    std::vector<float> l2_weights;
    int sequence_length = 0;
    int eos_token = -1;
    float repetition_scale = 0.0f;
    float l2_beta = 0.0f;
    float gradient_scale = 1.0f;
    int row_tile = 128;
    int vocabulary_tile = 256;
};
struct TiledCrossEntropyResult {
    Tensor losses; // [3]: CE, repetition unlikelihood, logit L2
    Tensor input_gradient;
    size_t maximum_logit_elements = 0;
};
namespace cce {
void mask_inactive_rows(Tensor& input, const Tensor& metadata);
void update_statistics(const Tensor& logits, Tensor& statistics,
                       const Tensor& metadata, int vocabulary_start);
Tensor finish_statistics(Tensor& statistics, const Tensor& metadata,
                         int vocabulary_size, float repetition_scale,
                         float l2_beta);
Tensor gradient_tile(const Tensor& logits, const Tensor& statistics,
                     const Tensor& metadata, int vocabulary_start,
                     int vocabulary_size, float repetition_scale,
                     float l2_beta, float gradient_scale);
void add_region(Tensor& destination, const Tensor& source,
                int row_start, int column_start);
} // namespace cce
} // namespace nsos
