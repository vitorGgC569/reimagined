#include "../include/jamba_utils.h"

#include <algorithm>

namespace nsos {

int sanitize_head_count(int d_model, int requested) {
    int heads = std::max(requested, 1);
    heads = std::min(heads, std::max(d_model, 1));
    while (heads > 1 && (d_model % heads) != 0) {
        --heads;
    }
    return std::max(heads, 1);
}

int select_kv_heads(int n_heads, int requested) {
    if (requested > 0) {
        int chosen = std::min(std::max(requested, 1), std::max(n_heads, 1));
        while (chosen > 1 && (n_heads % chosen) != 0) {
            --chosen;
        }
        return std::max(chosen, 1);
    }

    int chosen = std::max(n_heads / 4, 1);
    while (chosen > 1 && (n_heads % chosen) != 0) {
        --chosen;
    }
    return std::max(chosen, 1);
}

bool layer_matches_schedule(int layer_one_based, int period, int slot) {
    if (period <= 0) {
        return false;
    }
    const int normalized_slot = std::clamp(slot, 0, period - 1);
    return ((layer_one_based - 1) % period) == normalized_slot;
}

void prefix_parameter_names(std::vector<Parameter*>& params, const std::string& prefix) {
    for (auto* param : params) {
        if (!param) {
            continue;
        }
        const std::string current = param->name.empty() ? param->base_name : param->name;
        const std::string leaf = param->base_name.empty() ? current : param->base_name;
        param->base_name = leaf;
        param->name =
            (current.find('.') == std::string::npos) ? (prefix + leaf) : (prefix + current);
    }
}

Tensor make_zero_like(const Tensor& x) {
    return Tensor::zeros(x.shape.dims, x.get_device());
}

std::vector<int> normalize_valid_lengths(const std::vector<int>& lengths,
                                         int batch_size,
                                         int seq_len) {
    std::vector<int> normalized(static_cast<size_t>(batch_size), seq_len);
    if (static_cast<int>(lengths.size()) == batch_size) {
        for (int batch = 0; batch < batch_size; ++batch) {
            normalized[static_cast<size_t>(batch)] =
                std::clamp(lengths[static_cast<size_t>(batch)], 0, seq_len);
        }
    }
    return normalized;
}

} // namespace nsos
