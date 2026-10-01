#pragma once

#include "optimizer_runtime_policy.h"
#include "gpu_attention_training.h"

namespace nsos::training_policy {
inline bool head_cce() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_HEAD_CCE", false);
}
inline constexpr const char* head_cce_identity = "exact_unfiltered_bitlinear_row128_vocab256_recompute_v1";

// Experimental exact redesigns are explicit, validated policies. The same
// parser is used by dispatch and checkpoint identity; malformed values must
// never activate a path which the checkpoint fingerprint calls inactive.
inline bool boundary_history() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_MAMBA_BOUNDARY_HISTORY", false);
}
inline attention_training::Policy attention_provider() {
    return attention_training::policy();
}
inline bool tiled_attention() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_ATTN_TILED_TRAINING", false);
}
inline bool ordered_moe() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_MOE_ORDERED_DEVICE", false);
}
inline bool grouped_moe_training() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_MOE_GROUPED_TRAINING", false);
}
inline bool moe_wmma_training() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_MOE_WMMA_TRAINING", false);
}
inline constexpr const char* moe_wmma_identity = "rdna3_lowp_tile32_minrows16_mindim32_v1";
inline bool device_ttt() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_TTT_DEVICE_RECURRENCE", false);
}
inline bool full_ttt_bptt() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_TTT_FULL_BPTT", false);
}
inline bool kan_recompute_training() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_KAN_RECOMPUTE_TRAINING", false);
}
inline constexpr const char* kan_recompute_identity = "device_qat_tree256_rbf_tile16_recompute_v1";
inline bool kan_wmma_training() {
    return optimizer_policy::parse_optimizer_boolean("NSOS_KAN_WMMA_TRAINING", false);
}
inline constexpr const char* kan_wmma_identity = "rdna3_implicit_rbf_tile32_minrows16_mindim32_v1";

} // namespace nsos::training_policy
