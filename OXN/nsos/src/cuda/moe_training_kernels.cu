#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/kernels.cuh"
#include "cuda/moe_training_kernels.cuh"
#include "cuda/moe_training_wmma.cuh"
#include <cmath>
#include <algorithm>

using nsos::GpuMoeTrainingLinearView;
namespace {
constexpr int Tile = 16;
__global__ __launch_bounds__(256) void accumulate_gradients_kernel(
    const nsos::GpuMoeGradientView* views, const int* offsets,
    const float* weight, const float* bias, const float* magnitude,
    int experts, int elements, int outputs) {
    const int e = blockIdx.y;
    if (offsets[experts + 1] || offsets[e] == offsets[e + 1]) return;
    const auto view = views[e];
    const size_t total = static_cast<size_t>(elements) + 2u * outputs;
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * 256) {
        float* destination;
        const float* source;
        unsigned bit;
        size_t col = i;
        if (i < static_cast<size_t>(elements)) {
            destination = view.weight; source = weight + static_cast<size_t>(e) * elements; bit = 1;
        } else if (i < static_cast<size_t>(elements) + outputs) {
            col -= elements;
            destination = view.bias; source = bias + static_cast<size_t>(e) * outputs; bit = 2;
        } else {
            col -= static_cast<size_t>(elements) + outputs;
            destination = view.magnitude; source = magnitude + static_cast<size_t>(e) * outputs; bit = 4;
        }
        if (!destination) continue;
        const float value = source[col];
        if (view.add_mask & bit) destination[col] += value;
        else destination[col] = value;
    }
}
__global__ __launch_bounds__(256) void activity_accumulate_kernel(
    const nsos::GpuMoeGradientView* views, NsosSparseOptimizerActivity a,
    const float* weight, const float* bias, const float* magnitude,
    int elements, int outputs, float scale) {
    const int e = blockIdx.y;
    if (*a.abort_issue || !a.current[e]) return;
    const auto view = views[e];
    const bool first = a.first_write[e] != 0;
    const size_t total = static_cast<size_t>(elements) + 2u * outputs;
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * 256) {
        float* dst; const float* src; size_t col = i;
        if (i < static_cast<size_t>(elements)) {
            dst = view.weight; src = weight + static_cast<size_t>(e) * elements;
        } else if (i < static_cast<size_t>(elements) + outputs) {
            col -= elements; dst = view.bias; src = bias + static_cast<size_t>(e) * outputs;
        } else {
            col -= static_cast<size_t>(elements) + outputs;
            dst = view.magnitude; src = magnitude + static_cast<size_t>(e) * outputs;
        }
        if (!dst) continue;
        const float value = scale * src[col];
        if (first) dst[col] = value; else dst[col] += value;
    }
}
// Match the existing absmean block geometry, but write ordered partials rather
// than atomically accumulating them. Membership is consumed on the device;
// no latent weight from an inactive expert is fetched.
__global__ __launch_bounds__(256) void qat_partial_kernel(
    const GpuMoeTrainingLinearView* views, const int* offsets,
    float* partials, int experts, int elements) {
    const int e = blockIdx.y;
    if (offsets[experts + 1] || offsets[e] == offsets[e + 1]) return;
    const auto view = views[e];
    if (!view.qat_scale) return;
    __shared__ float tree[256];
    float sum = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
         i < static_cast<size_t>(elements); i += static_cast<size_t>(gridDim.x) * 256)
        sum += fabsf(view.latent_weight[i]);
    tree[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[static_cast<size_t>(e) * gridDim.x + blockIdx.x] = tree[0];
}
__global__ void qat_scale_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const float* partials, int experts, int elements, int blocks) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts || offsets[experts + 1] || offsets[e] == offsets[e + 1]) return;
    const auto view = views[e];
    if (!view.qat_scale) return;
    float sum = 0;
    for (int b = 0; b < blocks; ++b) sum += partials[static_cast<size_t>(e) * blocks + b];
    // These destinations are owned mutable workspaces, never master weights.
    const_cast<float*>(view.qat_scale)[0] = sum / static_cast<float>(elements) + 1e-8f;
}
__global__ void qat_quantize_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, int experts, int elements) {
    const int e = blockIdx.y;
    if (offsets[experts + 1] || offsets[e] == offsets[e + 1]) return;
    const auto view = views[e];
    if (!view.qat_scale) return;
    const float scale = view.qat_scale[0], inv = 1.0f / (scale + 1e-8f);
    float* output = const_cast<float*>(view.weight);
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
         i < static_cast<size_t>(elements); i += static_cast<size_t>(gridDim.x) * 256) {
        const float value = view.latent_weight[i] * inv;
        const float ternary = value > 0.5f ? 1.0f : value < -0.5f ? -1.0f : 0.0f;
        output[i] = ternary * scale;
    }
}
__global__ void qat_union_offsets_kernel(NsosSparseOptimizerActivity a, int* offsets) {
    if (threadIdx.x || blockIdx.x) return;
    offsets[0] = 0;
    for (int e = 0; e < a.experts; ++e) offsets[e+1] = offsets[e] + (a.accumulated[e] ? 1 : 0);
    offsets[a.experts+1] = *a.abort_issue;
}
__global__ void qat_union_regularize_kernel(const GpuMoeTrainingLinearView* views,
    const nsos::GpuMoeGradientView* gradients, NsosSparseOptimizerActivity a,
    double* partials, int elements, float base, int accumulation_steps) {
    const int e = blockIdx.y;
    const size_t partial = static_cast<size_t>(e)*gridDim.x + blockIdx.x;
    if (*a.abort_issue || !a.accumulated[e]) { if (!threadIdx.x) partials[partial]=0; return; }
    const auto view = views[e];
    if (!view.qat_scale) { if (!threadIdx.x) partials[partial]=0; return; }
    __shared__ double tree[256];
    double sum=0; auto* grad = gradients[e].weight;
    for (size_t i=static_cast<size_t>(blockIdx.x)*256+threadIdx.x;
         i<static_cast<size_t>(elements); i+=static_cast<size_t>(gridDim.x)*256) {
        const float diff=view.latent_weight[i]-view.weight[i];
        grad[i]+=(base*static_cast<float>(accumulation_steps))*diff;
        sum+=static_cast<double>(diff)*diff;
    }
    tree[threadIdx.x]=sum; __syncthreads();
    for(int stride=128;stride;stride>>=1) {if(threadIdx.x<stride)tree[threadIdx.x]+=tree[threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x)partials[partial]=tree[0];
}
__global__ void qat_loss_finalize_kernel(const double* partials, float* loss,
    NsosSparseOptimizerActivity a, int blocks, float base) {
    if(threadIdx.x || blockIdx.x || *a.abort_issue)return;
    double sum=0;
    for(int e=0;e<a.experts;++e)if(a.accumulated[e])
        for(int b=0;b<blocks;++b)sum+=partials[static_cast<size_t>(e)*blocks+b];
    *loss+=static_cast<float>(0.5*static_cast<double>(base)*sum);
}
__device__ int expert_for_slot(const int* offsets, int experts, int slot) {
    int lo = 0, hi = experts - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (slot >= offsets[mid + 1]) lo = mid + 1; else hi = mid;
    }
    return lo;
}
__device__ float operand(float value, int mode) {
    if (mode == 1) {
#if defined(NSOS_GPU_BACKEND_HIP)
        return static_cast<float>(hip_bfloat16(value));
#else
        return __bfloat162float(__float2bfloat16(value));
#endif
    }
    if (mode == 2) return __half2float(__float2half_rn(value));
    return value;
}

__global__ void validate_route_kernel(int* offsets, int rows, int experts, int capacity) {
    if (threadIdx.x || blockIdx.x) return;
    int issue = offsets[0] != 0 || offsets[experts] < 0 || offsets[experts] > capacity;
    for (int e = 0; e < experts; ++e)
        issue |= offsets[e] < 0 || offsets[e + 1] < offsets[e] ||
                 offsets[e + 1] - offsets[e] > rows;
    offsets[experts + 1] = issue;
}
__global__ void assign_kernel(const float* routing, const int* offsets,
    int* permutation, int* inverse, float* scales, int rows, int experts) {
    if (offsets[experts + 1]) return;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    int slot = offsets[e];
    for (int r = 0; r < rows; ++r) {
        const size_t i = static_cast<size_t>(r) * experts + e;
        const float weight = routing[i];
        inverse[i] = weight == 0 ? -1 : slot;
        if (weight != 0) { permutation[slot] = r; scales[slot++] = weight; }
    }
}

__global__ void prepare_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* source, bool gather,
    float* normalized, float* prepared, float* inverse_rms, int experts, int inputs) {
    const int slot = blockIdx.x;
    if (offsets[experts + 1] || slot >= offsets[experts]) return;
    const auto view = views[expert_for_slot(offsets, experts, slot)];
    const float* input = source + static_cast<size_t>(gather ? permutation[slot] : slot) * inputs;
    const size_t base = static_cast<size_t>(slot) * inputs;
    if (!view.rms_input && !view.activation_bits) {
        if (threadIdx.x == 0) inverse_rms[slot] = 1;
        for (int i = threadIdx.x; i < inputs; i += 256) {
            normalized[base + i] = input[i];
            if (prepared != normalized) prepared[base + i] = input[i];
        }
        return;
    }
    __shared__ float tree[256], inv, quant, dequant;
    float sum = 0;
    for (int i = threadIdx.x; i < inputs; i += 256) sum += input[i] * input[i];
    tree[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        inv = view.rms_input ? rsqrtf(tree[0] / inputs + 1e-6f) : 1;
        inverse_rms[slot] = inv;
    }
    __syncthreads();
    float maximum = 0;
    for (int i = threadIdx.x; i < inputs; i += 256) {
        const float value = input[i] * inv;
        normalized[base + i] = value;
        maximum = fmaxf(maximum, fabsf(value));
    }
    if (view.activation_bits) {
        tree[threadIdx.x] = maximum;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride) tree[threadIdx.x] = fmaxf(tree[threadIdx.x], tree[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float qmax = view.activation_bits <= 2 ? 1 : static_cast<float>((1 << (view.activation_bits - 1)) - 1);
            quant = qmax / (tree[0] + 1e-8f); dequant = (tree[0] + 1e-8f) / qmax;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < inputs; i += 256) {
        float value = normalized[base + i];
        if (view.activation_bits) {
            const float qmax = view.activation_bits <= 2 ? 1 : static_cast<float>((1 << (view.activation_bits - 1)) - 1);
            value = rintf(fminf(fmaxf(value * quant, -qmax), qmax)) * dequant;
        }
        prepared[base + i] = value;
    }
}

// FWD: input @ W^T; DX: grad_pre @ W; DW: grad_pre^T @ prepared.
// Fixed launch bounds, padded LDS, device segment sizes, single-owner outputs.
template<int Kind>
__global__ __launch_bounds__(256) void gemm_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const float* a, const float* b, float* output,
    float* post, float* squared, int experts, int inputs, int outputs, int mode) {
    const int e = blockIdx.z;
    if (offsets[experts + 1]) return;
    const int begin = offsets[e], count = offsets[e + 1] - begin;
    if (count == 0) return;
    const auto view = views[e];
    const int m = Kind == 2 ? outputs : count;
    const int n = Kind == 0 ? outputs : inputs;
    const int k = Kind == 0 ? inputs : Kind == 1 ? outputs : count;
    if (blockIdx.y * Tile >= m || blockIdx.x * Tile >= n) return;
    const int row = blockIdx.y * Tile + threadIdx.y;
    const int col = blockIdx.x * Tile + threadIdx.x;
    __shared__ float left[Tile][Tile + 1], right[Tile][Tile + 1];
    float sum = 0;
    for (int start = 0; start < k; start += Tile) {
        const int ak = start + threadIdx.x, bk = start + threadIdx.y;
        float av = 0, bv = 0;
        if constexpr (Kind == 0) {
            if (row < count && ak < inputs) av = a[static_cast<size_t>(begin + row) * inputs + ak];
            if (col < outputs && bk < inputs) bv = view.weight[static_cast<size_t>(col) * inputs + bk];
        } else if constexpr (Kind == 1) {
            if (row < count && ak < outputs) av = a[static_cast<size_t>(begin + row) * outputs + ak];
            if (col < inputs && bk < outputs) bv = view.weight[static_cast<size_t>(bk) * inputs + col];
        } else {
            if (row < outputs && ak < count) av = a[static_cast<size_t>(begin + ak) * outputs + row];
            if (col < inputs && bk < count) bv = b[static_cast<size_t>(begin + bk) * inputs + col];
        }
        left[threadIdx.y][threadIdx.x] = operand(av, mode);
        right[threadIdx.y][threadIdx.x] = operand(bv, mode);
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < Tile; ++t) sum = fmaf(left[threadIdx.y][t], right[t][threadIdx.x], sum);
        __syncthreads();
    }
    if (row >= m || col >= n) return;
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
            if (squared) { const float positive = fmaxf(0, value); squared[index] = positive * positive; }
        }
    }
}

__global__ void prepare_grad_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* scales,
    const float* upstream, bool gather, const float* squared_pre,
    float* grad_out, float* grad_pre, int capacity, int experts, int outputs) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(capacity) * outputs || offsets[experts + 1]) return;
    const int slot = index / outputs, col = index % outputs;
    if (slot >= offsets[experts]) return;
    const auto view = views[expert_for_slot(offsets, experts, slot)];
    float value = upstream[static_cast<size_t>(gather ? permutation[slot] : slot) * outputs + col];
    if (scales) value *= scales[slot];
    if (squared_pre) value = (value * 2.0f) * fmaxf(0, squared_pre[index]);
    grad_out[index] = value;
    grad_pre[index] = view.magnitude ? value * view.magnitude[col] : value;
}

__global__ void affine_grad_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const float* grad_out, const float* pre,
    float* grad_bias, float* grad_magnitude, int experts, int outputs) {
    const int e = blockIdx.z, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= outputs || offsets[experts + 1] || offsets[e] == offsets[e + 1]) return;
    const auto view = views[e];
    float bias_sum = 0, magnitude_sum = 0;
    for (int slot = offsets[e]; slot < offsets[e + 1]; ++slot) {
        const size_t i = static_cast<size_t>(slot) * outputs + col;
        if (view.bias) bias_sum += grad_out[i];
        if (view.magnitude) magnitude_sum += grad_out[i] * pre[i];
    }
    grad_bias[static_cast<size_t>(e) * outputs + col] = bias_sum;
    grad_magnitude[static_cast<size_t>(e) * outputs + col] = magnitude_sum;
}

__global__ void reverse_norm_kernel(const GpuMoeTrainingLinearView* views,
    const int* offsets, const float* normalized, const float* inverse_rms,
    float* grad, int experts, int inputs) {
    const int slot = blockIdx.x;
    if (offsets[experts + 1] || slot >= offsets[experts]) return;
    const auto view = views[expert_for_slot(offsets, experts, slot)];
    if (!view.rms_input) return;
    const size_t base = static_cast<size_t>(slot) * inputs;
    __shared__ float tree[256];
    float sum = 0;
    for (int i = threadIdx.x; i < inputs; i += 256) sum += grad[base + i] * normalized[base + i];
    tree[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = tree[0] / inputs, inv = inverse_rms[slot];
    for (int i = threadIdx.x; i < inputs; i += 256)
        grad[base + i] = (grad[base + i] - normalized[base + i] * mean) * inv;
}

__global__ void combine_kernel(const float* values, const int* inverse,
    const int* offsets, const float* scales, float* output, int rows, int dim, int experts) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * dim) return;
    if (offsets[experts + 1]) { output[i] = NAN; return; }
    const int row = i / dim, col = i % dim;
    float sum = 0;
    for (int e = 0; e < experts; ++e) {
        const int slot = inverse[static_cast<size_t>(row) * experts + e];
        if (slot >= 0) sum += values[static_cast<size_t>(slot) * dim + col] * (scales ? scales[slot] : 1);
    }
    output[i] = sum;
}
__global__ void router_grad_kernel(const float* grad, const float* expert_out,
    const int* inverse, const int* offsets, float* output, int rows, int dim, int experts) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * experts) return;
    if (offsets[experts + 1]) { output[i] = NAN; return; }
    const int slot = inverse[i];
    double sum = 0;
    if (slot >= 0) for (int d = 0; d < dim; ++d)
        sum += static_cast<double>(grad[(i / experts) * dim + d]) * expert_out[static_cast<size_t>(slot) * dim + d];
    output[i] = static_cast<float>(sum);
}
} // namespace

extern "C" bool launch_moe_training_accumulate_gradients(
    const nsos::GpuMoeGradientView* views, const int* offsets,
    const float* weight, const float* bias, const float* magnitude,
    int experts, int elements, int outputs) {
    if (!views || !offsets || !weight || !bias || !magnitude ||
        experts <= 0 || experts > 1024 || elements <= 0 || outputs <= 0) return false;
    const size_t total = static_cast<size_t>(elements) + 2u * outputs;
    const unsigned blocks = static_cast<unsigned>(total < 4096u * 256 ? (total + 255) / 256 : 4096);
    accumulate_gradients_kernel<<<dim3(blocks, experts), 256, 0, nsos::gpu::current_stream()>>>(
        views, offsets, weight, bias, magnitude, experts, elements, outputs);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_accumulate_gradients_activity(
    const nsos::GpuMoeGradientView* views, NsosSparseOptimizerActivity a,
    const float* weight, const float* bias, const float* magnitude, int elements, int outputs, float scale) {
    if (!views || !a.accumulated || !a.current || !a.first_write || !a.abort_issue ||
        !weight || !bias || !magnitude || a.experts <= 0 || a.experts > 1024 ||
        elements <= 0 || outputs <= 0 || !std::isfinite(scale) || scale <= 0) return false;
    const size_t total = static_cast<size_t>(elements) + 2u * outputs;
    const unsigned blocks = static_cast<unsigned>((std::min)(size_t(4096), (total + 255) / 256));
    activity_accumulate_kernel<<<dim3(blocks, a.experts), 256, 0, nsos::gpu::current_stream()>>>(
        views, a, weight, bias, magnitude, elements, outputs, scale);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_prepare_qat(const GpuMoeTrainingLinearView* views,
    const int* offsets, float* partials, int experts, int elements) {
    if (!views || !offsets || !partials || experts <= 0 || experts > 1024 || elements <= 0) return false;
    const int blocks = elements < 4096 * 256 ? (elements - 1) / 256 + 1 : 4096;
    const auto stream = nsos::gpu::current_stream();
    qat_partial_kernel<<<dim3(blocks, experts), 256, 0, stream>>>(views, offsets, partials, experts, elements);
    qat_scale_kernel<<<(experts + 255) / 256, 256, 0, stream>>>(views, offsets, partials, experts, elements, blocks);
    qat_quantize_kernel<<<dim3(blocks, experts), 256, 0, stream>>>(views, offsets, experts, elements);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_active_qat_regularization(
    const GpuMoeTrainingLinearView* views, const nsos::GpuMoeGradientView* gradients,
    NsosSparseOptimizerActivity a, int* union_offsets, float* partials, double* loss_partials, float* loss,
    int elements, float base, int accumulation_steps) {
    if (!views || !gradients || !a.accumulated || !a.abort_issue || a.experts <= 0 || a.experts > 1024 ||
        !union_offsets || !partials || !loss_partials || !loss || elements <= 0 || !std::isfinite(base) || base < 0 || accumulation_steps < 1) return false;
    const auto stream = nsos::gpu::current_stream();
    qat_union_offsets_kernel<<<1, 1, 0, stream>>>(a, union_offsets);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!launch_moe_training_prepare_qat(views, union_offsets, partials, a.experts, elements)) return false;
    const int blocks=(std::min)(4096,(elements-1)/256+1);
    qat_union_regularize_kernel<<<dim3(blocks,a.experts), 256, 0, stream>>>(views, gradients, a, loss_partials, elements, base, accumulation_steps);
    if(cudaGetLastError()!=cudaSuccess)return false;
    qat_loss_finalize_kernel<<<1,1,0,stream>>>(loss_partials,loss,a,blocks,base);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_route(const float* routing, int* counts, int* offsets,
    int* permutation, int* inverse, float* scales, int rows, int experts, int capacity) {
    if (!routing || !counts || !offsets || !permutation || !inverse || !scales ||
        rows <= 0 || experts <= 0 || experts > 1024 || capacity <= 0) return false;
    const auto stream = nsos::gpu::current_stream();
    if (cudaMemsetAsync(counts, 0, static_cast<size_t>(experts) * sizeof(int), stream) != cudaSuccess) return false;
    launch_moe_count_per_expert_kernel(routing, counts, rows, experts);
    launch_moe_exclusive_scan_small_kernel(counts, offsets, experts);
    validate_route_kernel<<<1, 1, 0, stream>>>(offsets, rows, experts, capacity);
    assign_kernel<<<(experts + 255) / 256, 256, 0, stream>>>(routing, offsets, permutation, inverse, scales, rows, experts);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_linear_forward(const GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* input, bool gather,
    float* normalized, float* prepared, float* inverse_rms, float* pre, float* post,
    float* squared_output, int rows, int capacity, int experts, int inputs, int outputs, int mode, bool use_wmma) {
    if (!views || !offsets || !permutation || !input || !normalized || !prepared ||
        !inverse_rms || !pre || !post || rows <= 0 || capacity <= 0 || experts <= 0 ||
        inputs <= 0 || outputs <= 0 || mode < 0 || mode > 2) return false;
    const auto stream = nsos::gpu::current_stream();
    prepare_kernel<<<capacity, 256, 0, stream>>>(views, offsets, permutation, input, gather,
        normalized, prepared, inverse_rms, experts, inputs);
    if (use_wmma && moe_training_wmma_geometry(rows, inputs, outputs, mode)) {
        if (!launch_moe_training_wmma_gemm(0, views, offsets, prepared, nullptr, pre, post,
            squared_output, rows, experts, inputs, outputs, mode)) return false;
    } else gemm_kernel<0><<<dim3((outputs + 15) / 16, (rows + 15) / 16, experts), dim3(16, 16), 0, stream>>>(
        views, offsets, prepared, nullptr, pre, post, squared_output, experts, inputs, outputs, mode);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_linear_backward(const GpuMoeTrainingLinearView* views,
    const int* offsets, const int* permutation, const float* scales,
    const float* upstream, bool gather, const float* squared_pre,
    const float* normalized, const float* prepared, const float* inverse_rms,
    const float* pre, float* grad_out, float* grad_pre, float* grad_input,
    float* grad_weight, float* grad_bias, float* grad_magnitude,
    int rows, int capacity, int experts, int inputs, int outputs, int mode, bool use_wmma) {
    if (!views || !offsets || !permutation || !upstream || !normalized || !prepared || !inverse_rms ||
        !pre || !grad_out || !grad_pre || !grad_input || !grad_weight || !grad_bias || !grad_magnitude ||
        rows <= 0 || capacity <= 0 || experts <= 0 || inputs <= 0 || outputs <= 0 || mode < 0 || mode > 2) return false;
    const auto stream = nsos::gpu::current_stream();
    const size_t elements = static_cast<size_t>(capacity) * outputs;
    prepare_grad_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(views, offsets, permutation, scales,
        upstream, gather, squared_pre, grad_out, grad_pre, capacity, experts, outputs);
    affine_grad_kernel<<<dim3((outputs + 255) / 256, 1, experts), 256, 0, stream>>>(
        views, offsets, grad_out, pre, grad_bias, grad_magnitude, experts, outputs);
    if (use_wmma && moe_training_wmma_geometry(rows, inputs, outputs, mode)) {
        if (!launch_moe_training_wmma_gemm(2, views, offsets, grad_pre, prepared, grad_weight, nullptr,
            nullptr, rows, experts, inputs, outputs, mode) ||
            !launch_moe_training_wmma_gemm(1, views, offsets, grad_pre, nullptr, grad_input, nullptr,
            nullptr, rows, experts, inputs, outputs, mode)) return false;
    } else {
    gemm_kernel<2><<<dim3((inputs + 15) / 16, (outputs + 15) / 16, experts), dim3(16, 16), 0, stream>>>(
        views, offsets, grad_pre, prepared, grad_weight, nullptr, nullptr, experts, inputs, outputs, mode);
    gemm_kernel<1><<<dim3((inputs + 15) / 16, (rows + 15) / 16, experts), dim3(16, 16), 0, stream>>>(
        views, offsets, grad_pre, nullptr, grad_input, nullptr, nullptr, experts, inputs, outputs, mode);
    }
    reverse_norm_kernel<<<capacity, 256, 0, stream>>>(views, offsets, normalized, inverse_rms, grad_input, experts, inputs);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_moe_training_combine(const float* values, const int* inverse,
    const int* offsets, const float* scales, float* output, int rows, int dim, int experts) {
    if (!values || !inverse || !offsets || !output || rows <= 0 || dim <= 0 || experts <= 0) return false;
    const size_t cells = static_cast<size_t>(rows) * dim;
    combine_kernel<<<(cells + 255) / 256, 256, 0, nsos::gpu::current_stream()>>>(
        values, inverse, offsets, scales, output, rows, dim, experts);
    return cudaGetLastError() == cudaSuccess;
}
extern "C" bool launch_moe_training_router_grad(const float* grad, const float* expert_out,
    const int* inverse, const int* offsets, float* router_grad, int rows, int dim, int experts) {
    if (!grad || !expert_out || !inverse || !offsets || !router_grad || rows <= 0 || dim <= 0 || experts <= 0) return false;
    const size_t cells = static_cast<size_t>(rows) * experts;
    router_grad_kernel<<<(cells + 255) / 256, 256, 0, nsos::gpu::current_stream()>>>(
        grad, expert_out, inverse, offsets, router_grad, rows, dim, experts);
    return cudaGetLastError() == cudaSuccess;
}
