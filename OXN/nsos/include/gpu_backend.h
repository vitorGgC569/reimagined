#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

// NSOS keeps one set of GPU sources for both supported compiler stacks.
// Existing implementation files still use the CUDA spellings internally;
// this header maps those spellings to HIP when NSOS_GPU_BACKEND_HIP is
// selected.  New public code should prefer the neutral helpers in namespace
// nsos::gpu and should not include vendor headers directly.

#if defined(NSOS_GPU_BACKEND_CUDA) && defined(NSOS_GPU_BACKEND_HIP)
#error "NSOS GPU backends are mutually exclusive"
#endif

#if defined(USE_CUDA) && \
    !defined(NSOS_GPU_BACKEND_CUDA) && !defined(NSOS_GPU_BACKEND_HIP)
// Backward compatibility for downstream builds that historically defined
// USE_CUDA themselves.
#define NSOS_GPU_BACKEND_CUDA 1
#endif

#if defined(NSOS_GPU_BACKEND_CUDA)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define NSOS_GPU_BACKEND_NAME "cuda"
#define NSOS_GPU_VENDOR_NAME "nvidia"

#elif defined(NSOS_GPU_BACKEND_HIP)

// HIPBLAS_V2 makes the Ex GEMM compute-type API match current cuBLAS more
// closely and is the supported hipBLAS interface for new applications.
#ifndef HIPBLAS_V2
#define HIPBLAS_V2
#endif

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#if defined(NSOS_INCLUDE_ROCWMMA) && defined(NSOS_HAS_ROCWMMA)
#include <rocwmma/rocwmma.hpp>
#endif

#define NSOS_GPU_BACKEND_NAME "hip"
#define NSOS_GPU_VENDOR_NAME "amd"

// Runtime types and values.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaDeviceProp hipDeviceProp_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaMemcpyKind hipMemcpyKind
#define cudaDataType_t hipDataType

#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeManaged hipMemoryTypeManaged
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaDevAttrComputeCapabilityMajor \
  hipDeviceAttributeComputeCapabilityMajor

// Runtime functions.
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemAdvise hipMemAdvise
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
inline hipError_t nsosHipGraphInstantiate(hipGraphExec_t* graph_exec,
                                          hipGraph_t graph,
                                          unsigned long long) {
  return hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
}
#define cudaGraphInstantiate nsosHipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy

// BLAS types, values and functions.
#define cublasStatus_t hipblasStatus_t
#define cublasHandle_t hipblasHandle_t
#define cublasComputeType_t hipblasComputeType_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F
#define __nv_bfloat16 hip_bfloat16
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGetVersion hipblasGetVersion
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx

// Keep the two compatibility checks in tensor.cpp on their established CUDA
// API branches. HIP pointer attributes expose `type`, and hipMemAdvise on
// Windows/Linux uses an integer device id.
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __builtin_nanf("")
#endif

// CUDA's masked shuffle is a warp-32 primitive. HIP supplies an unmasked
// shuffle whose width follows the native compiled wavefront. Device reduction
// code must therefore iterate over `warpSize`, which keeps this compatibility
// mapping correct for both AMD wave32 and wave64 without interpreting CUDA's
// 32-bit active mask as a wave64 mask.
#ifndef __shfl_down_sync
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta))
#endif

// HIP's signed dot API lowers through OCKL to the architecture's integer dot
// implementation. Keep the scalar oracle for non-gfx11 targets/toolchains.
static __device__ __forceinline__ int nsos_hip_dp4a(int lhs, int rhs, int acc) {
  const unsigned int a = static_cast<unsigned int>(lhs);
  const unsigned int b = static_cast<unsigned int>(rhs);
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && \
    !defined(NSOS_HIP_SCALAR_DP4A) && \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
  const char4 av = make_char4(static_cast<char>(a), static_cast<char>(a >> 8),
                             static_cast<char>(a >> 16), static_cast<char>(a >> 24));
  const char4 bv = make_char4(static_cast<char>(b), static_cast<char>(b >> 8),
                             static_cast<char>(b >> 16), static_cast<char>(b >> 24));
  return amd_mixed_dot(av, bv, acc, false);
#else
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const int av = static_cast<int>(
        static_cast<signed char>((a >> (byte * 8)) & 0xffu));
    const int bv = static_cast<int>(
        static_cast<signed char>((b >> (byte * 8)) & 0xffu));
    acc += av * bv;
  }
  return acc;
#endif
}
#ifndef __dp4a
#define __dp4a nsos_hip_dp4a
#endif

#else

#define NSOS_GPU_BACKEND_NAME "none"
#define NSOS_GPU_VENDOR_NAME "none"

#endif

namespace nsos::gpu {

#ifdef USE_CUDA
// All wrappers and library handles enqueue into the currently bound lane.
cudaStream_t current_stream() noexcept;
const int* decode_position() noexcept;
void set_decode_position(const int* position) noexcept;
#endif

enum class Backend {
  None,
  CUDA,
  HIP,
};

constexpr Backend compiled_backend() noexcept {
#if defined(NSOS_GPU_BACKEND_CUDA)
  return Backend::CUDA;
#elif defined(NSOS_GPU_BACKEND_HIP)
  return Backend::HIP;
#else
  return Backend::None;
#endif
}

constexpr const char* backend_name() noexcept {
  return NSOS_GPU_BACKEND_NAME;
}

constexpr const char* vendor_name() noexcept {
  return NSOS_GPU_VENDOR_NAME;
}

constexpr bool enabled() noexcept {
  return compiled_backend() != Backend::None;
}

#ifdef USE_CUDA
// Cleanup runs from destructors and exception handlers, where throwing would
// terminate the process or mask the primary failure. It must nevertheless be
// observable: no runtime release error is intentionally discarded.
inline void report_cleanup_status(cudaError_t status,
                                  const char* operation) noexcept {
  if (status == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "[GPU] cleanup failed during %s on %s: %s\n",
               operation != nullptr ? operation : "unknown operation",
               NSOS_GPU_BACKEND_NAME, cudaGetErrorString(status));
  (void)cudaGetLastError();
}
#endif

// Overflow-safe host-side grid arithmetic. The usual
// `(numerator + denominator - 1) / denominator` form is undefined for signed
// values near INT_MAX and wraps for unsigned values near their limit.
template <typename Numerator, typename Denominator>
constexpr std::common_type_t<Numerator, Denominator> ceil_div_positive(
    Numerator numerator, Denominator denominator) noexcept {
  using Common = std::common_type_t<Numerator, Denominator>;
  const Common n = static_cast<Common>(numerator);
  const Common d = static_cast<Common>(denominator);
  return n / d + static_cast<Common>((n % d) != 0);
}

struct DeviceInfo {
  int index = -1;
  std::string name;
  std::string architecture;
  std::size_t total_memory = 0;
  int warp_size = 0;
  bool integrated = false;
  bool compiled = false;
  bool fp16 = false;
  bool bf16 = false;
};

std::vector<DeviceInfo> enumerate_devices();

// Honors NSOS_GPU_DEVICE when set. Otherwise selects the discrete device with
// the most VRAM. Integrated GPUs are fail-closed unless
// NSOS_ALLOW_INTEGRATED_GPU=1 is explicitly set.
bool select_preferred_device(int* selected_device = nullptr,
                             std::string* error = nullptr);

}  // namespace nsos::gpu
