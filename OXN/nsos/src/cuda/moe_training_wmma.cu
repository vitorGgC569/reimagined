// RDNA3 grouped training GEMM. rocWMMA owns the architecture-specific register
// layout; the epilogue reads a documented store, never guessed fragment lanes.
#define NSOS_INCLUDE_ROCWMMA 1
#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/moe_training_wmma.cuh"

bool moe_training_wmma_supported() {
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) return false;
    thread_local int cached_device = -1;
    thread_local bool supported = false;
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        const std::string arch = std::string(prop.gcnArchName).substr(0, std::string(prop.gcnArchName).find(':'));
        supported = prop.warpSize == 32 && (arch == "gfx1100" || arch == "gfx1101" || arch == "gfx1102");
        bool compiled = false;
        for (const auto& info : nsos::gpu::enumerate_devices())
            if (info.index == device) compiled = info.compiled;
        supported = supported && compiled;
        cached_device = device;
    }
    return supported;
#else
    return false;
#endif
}

#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
namespace {
constexpr int Tile = 32;
template<int Kind, class DataT>
__global__ __launch_bounds__(128) void wmma_gemm(
    const nsos::GpuMoeTrainingLinearView* views, const int* offsets,
    const float* a, const float* b, float* output, float* post, float* squared,
    int experts, int inputs, int outputs) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    const int e = blockIdx.z;
    if (offsets[experts + 1]) return;
    const int begin = offsets[e], count = offsets[e + 1] - begin;
    const int m = Kind == 2 ? outputs : count;
    const int n = Kind == 0 ? outputs : inputs;
    const int k = Kind == 0 ? inputs : Kind == 1 ? outputs : count;
    const int row_base = blockIdx.y * Tile, col_base = blockIdx.x * Tile;
    // These branches are uniform over the entire CTA; no partial barrier exit.
    if (count <= 0 || row_base >= m || col_base >= n) return;
    const auto view = views[e];
    __shared__ __align__(32) DataT left[Tile * Tile], right[Tile * Tile];
    __shared__ __align__(32) float result[Tile * Tile];
    const int wave = threadIdx.x / 32;
    const int wr = (wave / 2) * 16, wc = (wave % 2) * 16;
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, DataT, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, DataT, rocwmma::row_major> fb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc;
    rocwmma::fill_fragment(acc, 0.0f);
    for (int start = 0; start < k; start += Tile) {
        for (int i = threadIdx.x; i < Tile * Tile; i += 128) {
            const int r = i / Tile, c = i % Tile;
            float av = 0, bv = 0;
            if constexpr (Kind == 0) {
                if (row_base + r < count && start + c < inputs)
                    av = a[static_cast<size_t>(begin + row_base + r) * inputs + start + c];
                // Fetch W rows coalesced, transpose only in LDS for A @ W^T.
                if (col_base + r < outputs && start + c < inputs)
                    bv = view.weight[static_cast<size_t>(col_base + r) * inputs + start + c];
            } else if constexpr (Kind == 1) {
                if (row_base + r < count && start + c < outputs)
                    av = a[static_cast<size_t>(begin + row_base + r) * outputs + start + c];
                if (col_base + c < inputs && start + r < outputs)
                    bv = view.weight[static_cast<size_t>(start + r) * inputs + col_base + c];
            } else {
                // dW consumes grad^T; transpose coalesced grad loads in LDS.
                if (row_base + c < outputs && start + r < count)
                    av = a[static_cast<size_t>(begin + start + r) * outputs + row_base + c];
                if (col_base + c < inputs && start + r < count)
                    bv = b[static_cast<size_t>(begin + start + r) * inputs + col_base + c];
            }
            left[Kind == 2 ? c * Tile + r : i] = static_cast<DataT>(av);
            right[Kind == 0 ? c * Tile + r : i] = static_cast<DataT>(bv);
        }
        __syncthreads();
        #pragma unroll
        for (int sub = 0; sub < Tile; sub += 16) {
            rocwmma::load_matrix_sync(fa, left + wr * Tile + sub, Tile);
            rocwmma::load_matrix_sync(fb, right + sub * Tile + wc, Tile);
            rocwmma::mma_sync(acc, fa, fb, acc);
        }
        __syncthreads();
    }
    rocwmma::store_matrix_sync(result + wr * Tile + wc, acc, Tile, rocwmma::layout_t::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < Tile * Tile; i += 128) {
        const int row = row_base + i / Tile, col = col_base + i % Tile;
        if (row >= m || col >= n) continue;
        float sum = result[i];
        if constexpr (Kind == 2) {
            const size_t wi = static_cast<size_t>(row) * inputs + col;
            if (view.qat_scale && fabsf(view.latent_weight[wi] * (1.0f / (view.qat_scale[0] + 1e-8f))) > 1)
                sum = 0;
            output[static_cast<size_t>(e) * outputs * inputs + wi] = sum;
        } else {
            const size_t index = static_cast<size_t>(begin + row) * n + col;
            output[index] = sum;
            if constexpr (Kind == 0) {
                float value = view.magnitude ? sum * view.magnitude[col] : sum;
                if (view.bias) value += view.bias[col];
                post[index] = value;
                if (squared) squared[index] = value > 0 ? value * value : 0;
            }
        }
    }
#endif
}
template<class DataT>
void launch(int kind, const nsos::GpuMoeTrainingLinearView* views, const int* offsets,
    const float* a, const float* b, float* output, float* post, float* squared,
    int rows, int experts, int inputs, int outputs) {
    const auto stream = nsos::gpu::current_stream();
    const int m = kind == 2 ? outputs : rows, n = kind == 0 ? outputs : inputs;
    const dim3 grid((n + Tile - 1) / Tile, (m + Tile - 1) / Tile, experts);
    if (kind == 0) wmma_gemm<0, DataT><<<grid, 128, 0, stream>>>(views, offsets, a, b, output, post, squared, experts, inputs, outputs);
    else if (kind == 1) wmma_gemm<1, DataT><<<grid, 128, 0, stream>>>(views, offsets, a, b, output, post, squared, experts, inputs, outputs);
    else wmma_gemm<2, DataT><<<grid, 128, 0, stream>>>(views, offsets, a, b, output, post, squared, experts, inputs, outputs);
}
} // namespace
#endif

bool launch_moe_training_wmma_gemm(int kind,
    const nsos::GpuMoeTrainingLinearView* views, const int* offsets,
    const float* a, const float* b, float* output, float* post, float* squared,
    int rows, int experts, int inputs, int outputs, int mode) {
    if (kind < 0 || kind > 2 || !views || !offsets || !a || !output ||
        (kind == 0 && !post) || (kind == 2 && !b) || experts <= 0 || experts > 1024 ||
        rows > 65535 * 32 || inputs > 65535 * 32 || outputs > 65535 * 32 ||
        !moe_training_wmma_geometry(rows, inputs, outputs, mode) || !moe_training_wmma_supported()) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    if (mode == 1) launch<rocwmma::bfloat16_t>(kind, views, offsets, a, b, output, post, squared, rows, experts, inputs, outputs);
    else launch<rocwmma::float16_t>(kind, views, offsets, a, b, output, post, squared, rows, experts, inputs, outputs);
    nsos::gpu::record_dispatch(nsos::gpu::DispatchPath::GroupedMoeWmmaGemm);
    return hipGetLastError() == hipSuccess;
#else
    return false;
#endif
}
