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

// Época global de nomeação de parâmetros.  JambaModel::parameters() chama
// bump_parameter_name_epoch() no início de cada passada; o primeiro
// prefix_parameter_names que toca um Parameter na época nova reconstrói o
// nome a partir do base_name (folha), tornando a nomeação idempotente entre
// passadas (antes, cada passada re-prefixava o absoluto da anterior).
void bump_parameter_name_epoch();
long long current_parameter_name_epoch();
Tensor make_zero_like(const Tensor& x);
std::vector<int> normalize_valid_lengths(const std::vector<int>& lengths,
                                         int batch_size,
                                         int seq_len);

} // namespace nsos
