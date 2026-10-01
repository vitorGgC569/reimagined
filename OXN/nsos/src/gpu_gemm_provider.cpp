#include "../include/gpu_gemm_provider.h"

#include "../include/gpu_backend.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nsos::gpu {
namespace {

std::string requested_provider() {
    const char* value = std::getenv("NSOS_GEMM_PROVIDER");
    if (value == nullptr || *value == '\0' ||
        std::string(value) == "auto" ||
        std::string(value) == "classic") {
        return "classic";
    }
    if (std::string(value) == "lt") {
        return "lt";
    }
    throw std::invalid_argument(
        "NSOS_GEMM_PROVIDER must be auto, classic, or lt");
}

}  // namespace

GemmProviderPolicy gemm_provider_policy(bool gpu_execution) {
    if (!gpu_execution) {
        return {GemmProviderKind::CpuReference, "cpu_reference",
                "ordered_cpu_v1", false, false};
    }

    const bool hip = std::string_view(backend_name()) == "hip";
#if defined(NSOS_ENABLE_BLAS_LT_PROVIDER)
    constexpr bool kLtCompiled = true;
#else
    constexpr bool kLtCompiled = false;
#endif
    const std::string request = requested_provider();
    if (request == "lt") {
        if (!kLtCompiled) {
            throw std::runtime_error(
                std::string(hip ? "hipBLASLt" : "cuBLASLt") +
                " was requested but this NSOS binary has no promoted Lt "
                "provider; use NSOS_GEMM_PROVIDER=classic");
        }
        return {GemmProviderKind::LtBlas,
                hip ? "hipblaslt" : "cublaslt",
                "lt_shape_registry_v1", true, false};
    }
    return {GemmProviderKind::ClassicBlas,
            hip ? "hipblas" : "cublas",
            "classic_gemm_ex_default_tensor_op_v1", kLtCompiled, false};
}

void require_classic_gemm_provider() {
    const GemmProviderPolicy policy = gemm_provider_policy(true);
    if (policy.kind != GemmProviderKind::ClassicBlas) {
        throw std::runtime_error(
            "The selected GEMM provider has no classic BLAS dispatch");
    }
}

}  // namespace nsos::gpu
