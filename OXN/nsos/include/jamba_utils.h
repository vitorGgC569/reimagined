#pragma once

#include "autograd.h"
#include "tensor.h"

#include <string>
#include <vector>

namespace nsos {

int sanitize_head_count(int d_model, int requested);
int select_kv_heads(int n_heads, int requested);
bool layer_matches_schedule(int layer_one_based, int period, int slot);
void prefix_parameter_names(std::vector<Parameter*>& params, const std::string& prefix);
Tensor make_zero_like(const Tensor& x);
std::vector<int> normalize_valid_lengths(const std::vector<int>& lengths,
                                         int batch_size,
                                         int seq_len);

} // namespace nsos
