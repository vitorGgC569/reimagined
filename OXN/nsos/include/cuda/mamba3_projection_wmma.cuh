#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nsos::mamba3_projection {
enum class Policy : int { ExactFP32=0, BF16=1, FP16=2 };
enum class ExactAxis : int { None=0, OutputRows=1, OutputCols=2, Reduction=3 };
// The independent arithmetic opt-in does not change phase/DT/A/state precision.
// Backward GEMMs round BOTH operands, including upstream adjoints, to lowp.
// This is the mixed-precision training algorithm; it is not an exact FP32 VJP.
inline Policy policy() {
    const char* value=std::getenv("NSOS_MAMBA3_PROJECTION_PROVIDER");
    if(!value || !*value || std::strcmp(value,"exact_fp32")==0) return Policy::ExactFP32;
    if(std::strcmp(value,"rdna_bf16_v1")==0) return Policy::BF16;
    if(std::strcmp(value,"rdna_fp16_v1")==0) return Policy::FP16;
    throw std::invalid_argument("NSOS_MAMBA3_PROJECTION_PROVIDER must be exact_fp32, rdna_bf16_v1 or rdna_fp16_v1");
}
inline const char* identity(Policy mode) {
    switch(mode) {
    case Policy::ExactFP32: return "dense_exact_fp32_v1";
    case Policy::BF16: return "rdna3_wave32_tile32_bf16_operands_adjoint_fp32acc_controltail_fp32_v1";
    case Policy::FP16: return "rdna3_wave32_tile32_fp16_operands_adjoint_fp32acc_controltail_fp32_v1";
    }
    throw std::invalid_argument("Invalid Mamba3 projection policy");
}
inline bool geometry(int rows,int cols,int reduction) {
    constexpr std::uint64_t limit=2147483647ULL;
    return rows>0 && cols>0 && reduction>0 && reduction<=2147483615 && rows<=65535*32 && cols<=65535*32 &&
        std::uint64_t(rows)*cols<=limit && std::uint64_t(rows)*reduction<=limit &&
        std::uint64_t(cols)*reduction<=limit;
}
#ifdef USE_CUDA
bool supported(Policy mode);
// Row-major physical operands. Transposition applies before multiplication.
// Output overwrites [rows,cols], FP32 storage/accumulation, no hidden allocation,
// no normalization, bias, STE clipping, optimizer, host copy or stream switch.
// Optional exact suffix is excluded from lowp casts; its dot products use FP32
// inputs and accumulation. Output-axis suffixes overwrite; reduction suffix adds.
bool gemm(Policy mode,bool transpose_left,bool transpose_right,int rows,int cols,int reduction,
    const float* left,const float* right,float* output,ExactAxis exact_axis=ExactAxis::None,int exact_begin=0);
#endif
}
