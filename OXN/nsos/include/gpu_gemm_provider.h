#pragma once

#include <string>

namespace nsos::gpu {

enum class GemmProviderKind {
    CpuReference,
    ClassicBlas,
    LtBlas,
};

// Vendor-neutral GEMM policy boundary. Common tensor code consumes this
// contract and never selects hipBLAS/cuBLAS-specific providers directly.
// Lt is represented explicitly but remains fail-closed until its backend has
// been compiled and passed the per-shape promotion gates.
struct GemmProviderPolicy {
    GemmProviderKind kind = GemmProviderKind::CpuReference;
    std::string provider_name;
    std::string algorithm_policy;
    bool lt_compiled = false;
    bool lt_promoted = false;
};

GemmProviderPolicy gemm_provider_policy(bool gpu_execution);
void require_classic_gemm_provider();

}  // namespace nsos::gpu
