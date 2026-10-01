// fused_optimizer_kernels.cu — otimizador multi-tensor (item #4 da auditoria).
//
// ANTES: apply_optimizer_step emitia ~3-4 kernels POR PARÂMETRO por step
// (scale 1/acc, norm, [clip-scale], adamw) — ~1850-2450 launches/step para os
// ~614 parâmetros do v11; medido opt=226-490 ms/step na T4.
//
// AGORA (padrão Apex/PyTorch-foreach): o passo inteiro
// vira 2 kernels + 1 D2H de 4 bytes:
//   K1  multi_tensor_sqsum   — Σg² de TODOS os grads (parciais por bloco +
//                              atomicAdd; mesma classe de não-determinismo do
//                              norm_kernel existente — sem regressão).
//   K2  multi_tensor_adamw   — atualização AdamW com o scale de acumulação E
//                              o coeficiente de clip DOBRADOS no gscale:
//        g_eff = g * gscale, onde gscale = (1/acc) * clip_coeff
//        m = β1·m + (1-β1)·g_eff ; v = β2·v + (1-β2)·g_eff²
//        w -= lr·wd·w (se wd_flag) ; w -= lr·(m/bc1)/(√(v/bc2)+ε)
//      Algebricamente idêntico a scale→clip→adamw do caminho antigo
//      (norm(g·s) = s·norm(g); o clip antigo multiplicava o grad pelo coef
//      antes do adamw — aqui o produto entra direto).  Diferença observável
//      única: os tensores .grad ficam CRUS após o step (antes ficavam
//      escalados+clipados) — estado interno, documentado.
//
// Metadados (ponteiros w/g/m/v, offsets prefixos, flags de weight-decay)
// sobem em H2D somente quando a coorte muda. O caminho determinístico de
// produção usa chunks imutáveis: um bloco pertence a exatamente um tensor e
// elimina a busca binária por elemento. A geometria plana permanece disponível
// como rollback e referência bitwise.

#include "gpu_backend.h"
#include "cuda/kernels.cuh"
#include "cuda/sparse_optimizer_activity.cuh"
#include <cmath>
#if defined(NSOS_GPU_BACKEND_CUDA)
#include <device_launch_parameters.h>
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

static_assert(sizeof(NsosMultiTensorChunk) == 16,
              "Multi-tensor chunk ABI must remain 16 bytes");

struct FusedOptMeta {
    float* const* w;
    float* const* g;
    float* const* m;
    float* const* v;
    const unsigned long long* offsets;  // [n_tensors + 1] prefixo de elementos
    const unsigned char* wd_flags;      // [n_tensors]
    const float* lr_scales;             // [n_tensors], nullable for sqsum
    int n_tensors;
    unsigned long long total;
    NsosOptimizerContribution contribution{};
};

__device__ inline bool optimizer_active(NsosOptimizerContribution contribution,
                                        int tensor) {
    if (contribution.abort_issue && *contribution.abort_issue != 0) return false;
    if (!contribution.predicates) return true;
    const unsigned char* predicate = contribution.predicates[tensor];
    return !predicate || *predicate != 0;
}

inline FusedOptMeta activity_meta(NsosActivityAwareOptimizerDesc desc) {
    return {desc.w, desc.g, desc.m, desc.v, desc.offsets, desc.wd_flags,
            desc.learning_rates, desc.n_tensors, desc.total, desc.contribution};
}

__device__ inline int find_tensor(const unsigned long long* offsets,
                                  int n_tensors,
                                  unsigned long long i) {
    int lo = 0;
    int hi = n_tensors - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (offsets[mid] <= i) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

__global__ void multi_tensor_sqsum_kernel(float* __restrict__ accum,
                                          FusedOptMeta meta) {
    __shared__ float partial[kThreads];
    float local = 0.0f;
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < meta.total;
         i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int t = find_tensor(meta.offsets, meta.n_tensors, i);
        if (!optimizer_active(meta.contribution, t)) continue;
        const float g = meta.g[t][i - meta.offsets[t]];
        local += g * g;
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(accum, partial[0]);
    }
}

__global__ void multi_tensor_adamw_kernel(FusedOptMeta meta,
                                          const float* gradient_sq_sum,
                                          float accumulation_scale,
                                          float max_grad_norm,
                                          float beta1,
                                          float beta2,
                                          float bc1,
                                          float bc2,
                                          float lr,
                                          float eps,
                                          float weight_decay,
                                          int* found_nonfinite) {
    __shared__ float shared_gscale;
    __shared__ int shared_input_valid;
    if (threadIdx.x == 0) {
        const float raw_sq_sum = *gradient_sq_sum;
        const float total_norm =
            accumulation_scale *
            sqrtf(fmaxf(raw_sq_sum, 0.0f));
        const bool norm_valid = isfinite(raw_sq_sum) && raw_sq_sum >= 0.0f &&
                                isfinite(total_norm);
        shared_input_valid =
            (meta.contribution.abort_issue
                 ? *meta.contribution.abort_issue == 0
                 : *found_nonfinite == 0) && norm_valid;
        if (*found_nonfinite == 0 && !norm_valid) {
            // Bit 1 is reserved for failures discovered while producing the
            // update. Bit 0 is the deferred pre-update finite gate.
            atomicExch(found_nonfinite, 2);
        }
        const float clip =
            shared_input_valid && total_norm > max_grad_norm
                ? max_grad_norm / (total_norm + 1e-6f)
                : (shared_input_valid ? 1.0f : 0.0f);
        shared_gscale = accumulation_scale * clip;
    }
    __syncthreads();
    if (!shared_input_valid) {
        return;
    }
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < meta.total;
         i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int t = find_tensor(meta.offsets, meta.n_tensors, i);
        if (!optimizer_active(meta.contribution, t)) continue;
        const unsigned long long k = i - meta.offsets[t];
        const float g = meta.g[t][k] * shared_gscale;
        const float m = beta1 * meta.m[t][k] + (1.0f - beta1) * g;
        const float v = beta2 * meta.v[t][k] + (1.0f - beta2) * g * g;
        meta.m[t][k] = m;
        meta.v[t][k] = v;
        const float p_lr = lr * (meta.lr_scales ? meta.lr_scales[t] : 1.0f);
        float w = meta.w[t][k];
        if (meta.wd_flags[t]) {
            w -= p_lr * weight_decay * w;
        }
        const float updated =
            w - p_lr * (m / bc1) / (sqrtf(v / bc2) + eps);
        meta.w[t][k] = updated;
        if (!isfinite(m) || !isfinite(v) ||
            !isfinite(updated)) {
            atomicExch(found_nonfinite, 2);
        }
    }
}

__global__ void multi_tensor_adamw_update_deterministic_kernel(
    FusedOptMeta meta, float beta1, float beta2, float bc1, float bc2,
    float eps, float weight_decay, int* found_nonfinite) {
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) +
             threadIdx.x;
         i < meta.total;
         i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int tensor = find_tensor(meta.offsets, meta.n_tensors, i);
        if (!optimizer_active(meta.contribution, tensor)) continue;
        const unsigned long long element = i - meta.offsets[tensor];
        const float g = meta.g[tensor][element];
        const float m_new =
            beta1 * meta.m[tensor][element] + (1.0f - beta1) * g;
        const float v_new =
            beta2 * meta.v[tensor][element] + (1.0f - beta2) * g * g;
        meta.m[tensor][element] = m_new;
        meta.v[tensor][element] = v_new;

        const float m_hat = m_new / bc1;
        const float v_hat = v_new / bc2;
        const float learning_rate = meta.lr_scales[tensor];
        float updated_weight = meta.w[tensor][element];
        if (meta.wd_flags[tensor]) {
            updated_weight -=
                learning_rate * weight_decay * updated_weight;
        }
        updated_weight -=
            learning_rate * m_hat / (sqrtf(v_hat) + eps);
        meta.w[tensor][element] = updated_weight;
        if (!isfinite(m_new) || !isfinite(v_new) ||
            !isfinite(updated_weight)) {
            atomicExch(found_nonfinite, 1);
        }
    }
}

// One block owns one contiguous interval from exactly one tensor. Elementwise
// arithmetic is deliberately duplicated from the flat reference kernel above:
// only scheduling changes, never the FP32 expression tree.
__global__ void multi_tensor_adamw_update_deterministic_chunked_kernel(
    FusedOptMeta meta, const NsosMultiTensorChunk* chunks,
    int chunk_count, float beta1, float beta2, float bc1, float bc2,
    float eps, float weight_decay, int* found_nonfinite) {
    const int chunk_index = static_cast<int>(blockIdx.x);
    if (chunk_index >= chunk_count) return;

    const NsosMultiTensorChunk chunk = chunks[chunk_index];
    if (chunk.tensor_index >= static_cast<unsigned int>(meta.n_tensors) ||
        chunk.element_count == 0u) {
        if (threadIdx.x == 0) atomicExch(found_nonfinite, 1);
        return;
    }
    const int tensor = static_cast<int>(chunk.tensor_index);
    if (!optimizer_active(meta.contribution, tensor)) return;
    const unsigned long long tensor_elements =
        meta.offsets[tensor + 1] - meta.offsets[tensor];
    if (chunk.element_offset > tensor_elements ||
        static_cast<unsigned long long>(chunk.element_count) >
            tensor_elements - chunk.element_offset) {
        if (threadIdx.x == 0) atomicExch(found_nonfinite, 1);
        return;
    }

    for (unsigned long long local = static_cast<unsigned int>(threadIdx.x);
         local < chunk.element_count;
         local += static_cast<unsigned int>(blockDim.x)) {
        const unsigned long long element = chunk.element_offset + local;
        const float g = meta.g[tensor][element];
        const float m_new =
            beta1 * meta.m[tensor][element] + (1.0f - beta1) * g;
        const float v_new =
            beta2 * meta.v[tensor][element] + (1.0f - beta2) * g * g;
        meta.m[tensor][element] = m_new;
        meta.v[tensor][element] = v_new;

        const float m_hat = m_new / bc1;
        const float v_hat = v_new / bc2;
        const float learning_rate = meta.lr_scales[tensor];
        float updated_weight = meta.w[tensor][element];
        if (meta.wd_flags[tensor]) {
            updated_weight -=
                learning_rate * weight_decay * updated_weight;
        }
        updated_weight -=
            learning_rate * m_hat / (sqrtf(v_hat) + eps);
        meta.w[tensor][element] = updated_weight;
        if (!isfinite(m_new) || !isfinite(v_new) ||
            !isfinite(updated_weight)) {
            atomicExch(found_nonfinite, 1);
        }
    }
}

// One block owns one tensor.  This fixed association gives a deterministic
// reduction order and avoids both atomics and the old full-tensor D2H copies.
__global__ void multi_tensor_criticality_metrics_kernel(
    float* const* w, const unsigned long long* offsets, const int* fan_in,
    int n_tensors, float* gammas, float* gains) {
    const int t = static_cast<int>(blockIdx.x);
    if (t >= n_tensors) return;
    const unsigned long long n64 = offsets[t + 1] - offsets[t];
    if (n64 == 0) {
        if (threadIdx.x == 0) {
            gammas[t] = 0.0f;
            gains[t] = 0.0f;
        }
        return;
    }
    __shared__ float abs_partial[kThreads];
    __shared__ unsigned int zero_partial[kThreads];
    float local_abs = 0.0f;
    for (unsigned long long i = threadIdx.x; i < n64; i += blockDim.x) {
        local_abs += fabsf(w[t][i]);
    }
    abs_partial[threadIdx.x] = local_abs;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            abs_partial[threadIdx.x] += abs_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float gamma = abs_partial[0] / static_cast<float>(n64);
    unsigned int local_zeros = 0;
    const float threshold = 0.5f * gamma;
    for (unsigned long long i = threadIdx.x; i < n64; i += blockDim.x) {
        local_zeros += fabsf(w[t][i]) < threshold ? 1u : 0u;
    }
    zero_partial[threadIdx.x] = local_zeros;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            zero_partial[threadIdx.x] += zero_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float nonzero_fraction =
            1.0f - static_cast<float>(zero_partial[0]) /
                       static_cast<float>(n64);
        gammas[t] = gamma;
        gains[t] = gamma * gamma * nonzero_fraction *
                   static_cast<float>(fan_in[t]);
    }
}

__global__ void multi_tensor_criticality_grad_kernel(
    float* const* w, float* const* g, const unsigned long long* offsets,
    const float* coefficients, int n_tensors, unsigned long long total) {
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) +
             threadIdx.x;
         i < total;
         i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int t = find_tensor(offsets, n_tensors, i);
        const unsigned long long k = i - offsets[t];
        const float value = w[t][k];
        const float sign = value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
        g[t][k] += coefficients[t] * sign;
    }
}

__global__ void multi_tensor_check_finite_kernel(
    int* found_issue, const float* const* values,
    const unsigned long long* offsets, int n_tensors,
    unsigned long long total,
    const unsigned char* require_nonnegative,
    NsosOptimizerContribution contribution, const unsigned char* initialized) {
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) +
             threadIdx.x;
         i < total;
         i += static_cast<unsigned long long>(gridDim.x) *
              blockDim.x) {
        const int tensor = find_tensor(offsets, n_tensors, i);
        if (!optimizer_active(contribution, tensor)) continue;
        if (initialized && initialized[tensor] == 0) continue;
        const float value =
            values[tensor][i - offsets[tensor]];
        if (!isfinite(value) ||
            (require_nonnegative != nullptr &&
             require_nonnegative[tensor] != 0 && value < 0.0f)) {
            atomicExch(found_issue, 1);
            return;
        }
    }
}

__global__ void multi_tensor_check_finite_chunked_kernel(
    int* found_issue, const float* const* values,
    const unsigned long long* offsets, int n_tensors,
    const unsigned char* require_nonnegative,
    const NsosMultiTensorChunk* chunks, int chunk_count,
    NsosOptimizerContribution contribution, const unsigned char* initialized) {
    const int chunk_index = static_cast<int>(blockIdx.x);
    if (chunk_index >= chunk_count) return;
    const NsosMultiTensorChunk chunk = chunks[chunk_index];
    if (chunk.tensor_index >= static_cast<unsigned int>(n_tensors) ||
        chunk.element_count == 0u) {
        if (threadIdx.x == 0) atomicExch(found_issue, 1);
        return;
    }
    const int tensor = static_cast<int>(chunk.tensor_index);
    if (!optimizer_active(contribution, tensor)) return;
    if (initialized && initialized[tensor] == 0) return;
    const unsigned long long tensor_elements =
        offsets[tensor + 1] - offsets[tensor];
    if (chunk.element_offset > tensor_elements ||
        static_cast<unsigned long long>(chunk.element_count) >
            tensor_elements - chunk.element_offset) {
        if (threadIdx.x == 0) atomicExch(found_issue, 1);
        return;
    }
    const bool nonnegative =
        require_nonnegative != nullptr &&
        require_nonnegative[tensor] != 0;
    for (unsigned long long local = static_cast<unsigned int>(threadIdx.x);
         local < chunk.element_count;
         local += static_cast<unsigned int>(blockDim.x)) {
        const float value =
            values[tensor][chunk.element_offset + local];
        if (!isfinite(value) || (nonnegative && value < 0.0f)) {
            atomicExch(found_issue, 1);
            return;
        }
    }
}

__global__ void multi_tensor_zero_kernel(
    float* const* values, const unsigned long long* offsets,
    int n_tensors, unsigned long long total) {
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) +
             threadIdx.x;
         i < total;
         i += static_cast<unsigned long long>(gridDim.x) *
              blockDim.x) {
        const int tensor = find_tensor(offsets, n_tensors, i);
        values[tensor][i - offsets[tensor]] = 0.0f;
    }
}

inline int blocks_for_total(unsigned long long total) {
    const unsigned long long b = nsos::gpu::ceil_div_positive(
        total, static_cast<unsigned long long>(kThreads));
    return static_cast<int>(b < kMaxBlocks ? b : kMaxBlocks);
}

}  // namespace

extern "C" void launch_multi_tensor_sqsum(float* accum,
                                          float* const* w,
                                          float* const* g,
                                          float* const* m,
                                          float* const* v,
                                          const unsigned long long* offsets,
                                          const unsigned char* wd_flags,
                                          int n_tensors,
                                          unsigned long long total) {
    if (total == 0 || n_tensors <= 0) return;
    FusedOptMeta meta{w, g, m, v, offsets, wd_flags, nullptr, n_tensors, total};
    multi_tensor_sqsum_kernel<<<blocks_for_total(total), kThreads, 0, nsos::gpu::current_stream()>>>(accum, meta);
}

extern "C" void launch_multi_tensor_adamw(float* const* w,
                                          float* const* g,
                                          float* const* m,
                                          float* const* v,
                                          const unsigned long long* offsets,
                                          const unsigned char* wd_flags,
                                          const float* lr_scales,
                                          int n_tensors,
                                          unsigned long long total,
                                          const float* gradient_sq_sum,
                                          float accumulation_scale,
                                          float max_grad_norm,
                                          float beta1,
                                          float beta2,
                                          float bc1,
                                          float bc2,
                                          float lr,
                                          float eps,
                                          float weight_decay,
                                          int* found_nonfinite) {
    if (total == 0 || n_tensors <= 0) return;
    FusedOptMeta meta{w, g, m, v, offsets, wd_flags, lr_scales, n_tensors,
                      total};
    multi_tensor_adamw_kernel<<<blocks_for_total(total), kThreads, 0, nsos::gpu::current_stream()>>>(
        meta, gradient_sq_sum, accumulation_scale, max_grad_norm,
        beta1, beta2, bc1, bc2, lr, eps, weight_decay,
        found_nonfinite);
}

extern "C" bool launch_multi_tensor_adamw_update_deterministic(
    float* const* w, float* const* g, float* const* m, float* const* v,
    const unsigned long long* offsets, const unsigned char* wd_flags,
    const float* learning_rates, int n_tensors, unsigned long long total,
    const NsosMultiTensorChunk* chunks, int chunk_count,
    float beta1, float beta2, float bc1, float bc2, float eps,
    float weight_decay, int* found_nonfinite) {
    if (w == nullptr || g == nullptr || m == nullptr || v == nullptr ||
        offsets == nullptr || wd_flags == nullptr || total == 0 ||
        n_tensors <= 0 || learning_rates == nullptr || chunk_count < 0 ||
        ((chunks == nullptr) != (chunk_count == 0)) ||
        found_nonfinite == nullptr || !std::isfinite(beta1) ||
        !std::isfinite(beta2) || !std::isfinite(bc1) ||
        !std::isfinite(bc2) || !std::isfinite(eps) ||
        !std::isfinite(weight_decay) || beta1 < 0.0f || beta1 >= 1.0f ||
        beta2 < 0.0f || beta2 >= 1.0f || bc1 <= 0.0f || bc2 <= 0.0f ||
        eps <= 0.0f || weight_decay < 0.0f) {
        return false;
    }
    FusedOptMeta meta{w, g, m, v, offsets, wd_flags, learning_rates,
                      n_tensors, total};
    if (chunk_count > 0) {
        multi_tensor_adamw_update_deterministic_chunked_kernel
            <<<chunk_count, kThreads, 0, nsos::gpu::current_stream()>>>(
                meta, chunks, chunk_count, beta1, beta2, bc1, bc2,
                eps, weight_decay, found_nonfinite);
    } else {
        multi_tensor_adamw_update_deterministic_kernel
            <<<blocks_for_total(total), kThreads, 0, nsos::gpu::current_stream()>>>(
                meta, beta1, beta2, bc1, bc2, eps, weight_decay,
                found_nonfinite);
    }
    return true;
}

extern "C" bool launch_multi_tensor_criticality_metrics(
    float* const* w, const unsigned long long* offsets, const int* fan_in,
    int n_tensors, float* gammas, float* gains) {
    if (w == nullptr || offsets == nullptr || fan_in == nullptr ||
        gammas == nullptr || gains == nullptr || n_tensors <= 0) {
        return false;
    }
    multi_tensor_criticality_metrics_kernel<<<n_tensors, kThreads, 0, nsos::gpu::current_stream()>>>(
        w, offsets, fan_in, n_tensors, gammas, gains);
    return true;
}

extern "C" bool launch_multi_tensor_criticality_grad(
    float* const* w, float* const* g, const unsigned long long* offsets,
    const float* coefficients, int n_tensors, unsigned long long total) {
    if (w == nullptr || g == nullptr || offsets == nullptr ||
        coefficients == nullptr || n_tensors <= 0 || total == 0) {
        return false;
    }
    multi_tensor_criticality_grad_kernel<<<blocks_for_total(total), kThreads, 0, nsos::gpu::current_stream()>>>(
        w, g, offsets, coefficients, n_tensors, total);
    return true;
}

extern "C" bool launch_multi_tensor_check_finite(
    int* found_issue, const float* const* values,
    const unsigned long long* offsets, int n_tensors,
    unsigned long long total,
    const unsigned char* require_nonnegative,
    const NsosMultiTensorChunk* chunks, int chunk_count) {
    if (found_issue == nullptr || values == nullptr ||
        offsets == nullptr || n_tensors <= 0 || total == 0 ||
        chunk_count < 0 ||
        ((chunks == nullptr) != (chunk_count == 0))) {
        return false;
    }
    if (chunk_count > 0) {
        multi_tensor_check_finite_chunked_kernel<<<chunk_count, kThreads, 0, nsos::gpu::current_stream()>>>(
            found_issue, values, offsets, n_tensors,
            require_nonnegative, chunks, chunk_count, NsosOptimizerContribution{}, nullptr);
    } else {
        multi_tensor_check_finite_kernel<<<blocks_for_total(total),
                                           kThreads, 0, nsos::gpu::current_stream()>>>(
            found_issue, values, offsets, n_tensors, total,
            require_nonnegative, NsosOptimizerContribution{}, nullptr);
    }
    return true;
}

extern "C" bool launch_multi_tensor_zero(
    float* const* values, const unsigned long long* offsets,
    int n_tensors, unsigned long long total) {
    if (values == nullptr || offsets == nullptr ||
        n_tensors <= 0 || total == 0) {
        return false;
    }
    multi_tensor_zero_kernel<<<blocks_for_total(total), kThreads, 0, nsos::gpu::current_stream()>>>(
        values, offsets, n_tensors, total);
    return true;
}

namespace {

bool valid_activity(NsosSparseOptimizerActivity activity) {
    return activity.experts > 0 && activity.accumulated && activity.current &&
           activity.first_write && activity.abort_issue &&
           activity.accumulated != activity.current &&
           activity.accumulated != activity.first_write &&
           activity.current != activity.first_write;
}

__global__ void sparse_activity_reset_kernel(NsosSparseOptimizerActivity a) {
    for (int e = static_cast<int>(threadIdx.x); e < a.experts; e += blockDim.x) {
        a.accumulated[e] = 0;
        a.current[e] = 0;
        a.first_write[e] = 0;
    }
    if (threadIdx.x == 0) *a.abort_issue = 0;
}

// One ordered owner validates the complete bank before publication. No partial
// union is exposed if a later expert has an invalid offset.
__global__ void sparse_activity_validate_kernel(NsosSparseOptimizerActivity a,
    const int* offsets, int capacity, const int* routing_issue) {
    if (threadIdx.x != 0) return;
    if (*a.abort_issue != 0) return;
    bool valid = !routing_issue || *routing_issue == 0;
    valid = valid && offsets[0] == 0 && offsets[a.experts] >= 0 && offsets[a.experts] <= capacity;
    for (int e = 0; valid && e < a.experts; ++e) {
        valid = offsets[e] >= 0 && offsets[e] <= offsets[e + 1] &&
                offsets[e + 1] <= capacity;
    }
    if (!valid) *a.abort_issue = 1;
}

__global__ void sparse_activity_publish_kernel(NsosSparseOptimizerActivity a,
    const int* offsets, bool abort) {
    const bool valid = !abort && *a.abort_issue == 0;
    for (unsigned long long e = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         e < static_cast<unsigned long long>(a.experts);
         e += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const bool active = valid && offsets[e + 1] > offsets[e];
        a.current[e] = active ? 1 : 0;
        a.first_write[e] = active && a.accumulated[e] == 0 ? 1 : 0;
        if (active) a.accumulated[e] = 1;
    }
}

__global__ void sparse_activity_abort_kernel(int* issue) {
    if (threadIdx.x == 0) *issue = 1;
}

bool valid_desc(NsosActivityAwareOptimizerDesc d) {
    return d.offsets && d.n_tensors > 0 && d.total > 0;
}

bool valid_chunks(const NsosMultiTensorChunk* chunks, int count) {
    return count >= 0 && ((chunks == nullptr) == (count == 0));
}

bool valid_adam(float beta1, float beta2, float bc1, float bc2,
                float eps, float weight_decay) {
    return std::isfinite(beta1) && std::isfinite(beta2) &&
           std::isfinite(bc1) && std::isfinite(bc2) && std::isfinite(eps) &&
           std::isfinite(weight_decay) && beta1 >= 0 && beta1 < 1 &&
           beta2 >= 0 && beta2 < 1 && bc1 > 0 && bc2 > 0 &&
           eps > 0 && weight_decay >= 0;
}

__global__ void activity_preflight_kernel(int* issue, NsosActivityAwareOptimizerDesc d,
    const NsosMultiTensorChunk* chunks, int chunk_count) {
    if (threadIdx.x != 0 || *issue != 0) return;
    bool valid = d.offsets[0] == 0 && d.offsets[d.n_tensors] == d.total;
    for (int t = 0; valid && t < d.n_tensors; ++t) {
        valid = d.offsets[t] <= d.offsets[t + 1] && d.offsets[t + 1] <= d.total;
        if (valid && optimizer_active(d.contribution, t)) {
            valid = d.w[t] && d.g[t] && d.m[t] && d.v[t] &&
                    isfinite(d.learning_rates[t]) && d.learning_rates[t] >= 0;
        }
    }
    // Ordered, gap-free, non-overlapping coverage. Empty tensors need no chunk.
    int tensor = 0;
    unsigned long long element = 0;
    for (int c = 0; valid && c < chunk_count; ++c) {
        while (tensor < d.n_tensors && element == d.offsets[tensor + 1] - d.offsets[tensor]) {
            ++tensor;
            element = 0;
        }
        const auto chunk = chunks[c];
        valid = tensor < d.n_tensors && chunk.tensor_index == static_cast<unsigned int>(tensor) &&
                chunk.element_offset == element && chunk.element_count > 0;
        if (valid) {
            const auto length = d.offsets[tensor + 1] - d.offsets[tensor];
            valid = chunk.element_count <= length - element;
            if (valid) element += chunk.element_count;
        }
    }
    if (valid && chunk_count > 0) {
        while (tensor < d.n_tensors && element == d.offsets[tensor + 1] - d.offsets[tensor]) {
            ++tensor;
            element = 0;
        }
        valid = tensor == d.n_tensors;
    }
    if (!valid) *issue = 1;
}

__global__ void activity_scale_kernel(NsosActivityAwareOptimizerDesc d, float scale) {
    for (unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < d.total; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int t = find_tensor(d.offsets, d.n_tensors, i);
        if (!optimizer_active(d.contribution, t)) continue;
        d.g[t][i - d.offsets[t]] *= scale;
    }
}

__device__ bool activity_chunk_valid(NsosActivityAwareOptimizerDesc d,
    NsosMultiTensorChunk chunk, int* issue) {
    if (chunk.tensor_index >= static_cast<unsigned int>(d.n_tensors) ||
        chunk.element_count == 0) {
        atomicExch(issue, 1);
        return false;
    }
    const auto begin = d.offsets[chunk.tensor_index];
    const auto end = d.offsets[chunk.tensor_index + 1];
    if (end < begin || end > d.total || chunk.element_offset > end - begin ||
        chunk.element_count > end - begin - chunk.element_offset) {
        atomicExch(issue, 1);
        return false;
    }
    return true;
}

__global__ void activity_norm_partials_kernel(double* partials,
    int* issue, NsosActivityAwareOptimizerDesc d, const NsosMultiTensorChunk* chunks) {
    __shared__ double tree[kThreads];
    const auto chunk = chunks[blockIdx.x];
    const bool valid = activity_chunk_valid(d, chunk, issue);
    double local = 0;
    // No gradient pointer may be fetched before the contribution check.
    if (valid && optimizer_active(d.contribution, chunk.tensor_index)) {
        const float* grad = d.g[chunk.tensor_index];
        for (unsigned long long i = threadIdx.x; i < chunk.element_count; i += blockDim.x) {
            const double value = static_cast<double>(grad[chunk.element_offset + i]);
            local += value * value;
        }
    }
    tree[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = tree[0];
}

__global__ void activity_norm_finalize_kernel(double* total, const double* partials,
    float* coefficient, int* issue, NsosOptimizerContribution contribution,
    int count, float max_norm) {
    if (threadIdx.x != 0) return;
    double sum = 0;
    for (int i = 0; i < count; ++i) sum += partials[i];
    const float norm = static_cast<float>(sqrt(sum));
    if ((contribution.abort_issue && *contribution.abort_issue != 0) ||
        sum < 0 || !isfinite(sum) || !isfinite(norm)) *issue = 1;
    *total = *issue == 0 ? sum : NAN;
    *coefficient = *issue == 0
        ? (norm > max_norm ? max_norm / (norm + 1e-6f) : 1.0f) : 0.0f;
}

__global__ void activity_clip_kernel(NsosActivityAwareOptimizerDesc d,
    const NsosMultiTensorChunk* chunks, const float* coefficient, const int* issue) {
    if (*issue != 0 || *coefficient == 1.0f) return;
    const auto chunk = chunks[blockIdx.x];
    if (!optimizer_active(d.contribution, chunk.tensor_index)) return;
    float* grad = d.g[chunk.tensor_index];
    for (unsigned long long i = threadIdx.x; i < chunk.element_count; i += blockDim.x)
        grad[chunk.element_offset + i] *= *coefficient;
}

__global__ void activity_initialize_moments_kernel(NsosActivityAwareOptimizerDesc d,
    unsigned char* initialized) {
    const int t = static_cast<int>(blockIdx.x);
    if (!optimizer_active(d.contribution, t) || initialized[t] != 0) return;
    const auto n = d.offsets[t + 1] - d.offsets[t];
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
        d.m[t][i] = 0.0f;
        d.v[t][i] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) initialized[t] = 1;
}

}  // namespace

extern "C" bool launch_sparse_optimizer_activity_reset(NsosSparseOptimizerActivity a) {
    if (!valid_activity(a)) return false;
    sparse_activity_reset_kernel<<<1, kThreads, 0, nsos::gpu::current_stream()>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_sparse_optimizer_activity_update(NsosSparseOptimizerActivity a,
    const int* offsets, int capacity, const int* routing_issue) {
    if (!valid_activity(a) || !offsets || capacity < 0) return false;
    const auto stream = nsos::gpu::current_stream();
    sparse_activity_validate_kernel<<<1, 1, 0, stream>>>(a, offsets, capacity, routing_issue);
    if (cudaGetLastError() != cudaSuccess) return false;
    sparse_activity_publish_kernel<<<blocks_for_total(a.experts), kThreads, 0, stream>>>(a, offsets, false);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_sparse_optimizer_activity_abort(NsosSparseOptimizerActivity a) {
    if (!valid_activity(a)) return false;
    const auto stream = nsos::gpu::current_stream();
    sparse_activity_abort_kernel<<<1, 1, 0, stream>>>(a.abort_issue);
    if (cudaGetLastError() != cudaSuccess) return false;
    sparse_activity_publish_kernel<<<blocks_for_total(a.experts), kThreads, 0, stream>>>(a, nullptr, true);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_sqsum(float* accum,
    NsosActivityAwareOptimizerDesc d) {
    if (!valid_desc(d) || !accum || !d.g) return false;
    multi_tensor_sqsum_kernel<<<blocks_for_total(d.total), kThreads, 0, nsos::gpu::current_stream()>>>(accum, activity_meta(d));
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_preflight(int* found_issue,
    NsosActivityAwareOptimizerDesc d, const NsosMultiTensorChunk* chunks, int chunk_count) {
    if (!valid_desc(d) || !d.w || !d.g || !d.m || !d.v || !d.learning_rates ||
        !d.wd_flags || !found_issue || d.contribution.abort_issue != found_issue ||
        !valid_chunks(chunks, chunk_count)) return false;
    activity_preflight_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(found_issue, d, chunks, chunk_count);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_scale_gradients(
    NsosActivityAwareOptimizerDesc d, float scale) {
    if (!valid_desc(d) || !d.g || !std::isfinite(scale) || scale < 0) return false;
    if (scale == 1.0f) return true;
    activity_scale_kernel<<<blocks_for_total(d.total), kThreads, 0, nsos::gpu::current_stream()>>>(d, scale);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_check_finite(int* found_issue,
    const float* const* values, const unsigned char* require_nonnegative,
    NsosActivityAwareOptimizerDesc d, const NsosMultiTensorChunk* chunks, int chunk_count,
    const unsigned char* initialized) {
    if (!valid_desc(d) || !found_issue || !values || !valid_chunks(chunks, chunk_count)) return false;
    if (chunk_count > 0) {
        multi_tensor_check_finite_chunked_kernel<<<chunk_count, kThreads, 0, nsos::gpu::current_stream()>>>(
            found_issue, values, d.offsets, d.n_tensors, require_nonnegative,
            chunks, chunk_count, d.contribution, initialized);
    } else {
        multi_tensor_check_finite_kernel<<<blocks_for_total(d.total), kThreads, 0, nsos::gpu::current_stream()>>>(
            found_issue, values, d.offsets, d.n_tensors, d.total,
            require_nonnegative, d.contribution, initialized);
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_chunked_norm_device_clip(double* total, double* partials,
    float* coefficient, int* found_issue, NsosActivityAwareOptimizerDesc d,
    const NsosMultiTensorChunk* chunks, int chunk_count, float max_norm) {
    if (!valid_desc(d) || !d.g || !total || !partials || !coefficient ||
        !found_issue || !chunks || chunk_count <= 0 || !std::isfinite(max_norm) || max_norm <= 0) return false;
    const auto stream = nsos::gpu::current_stream();
    activity_norm_partials_kernel<<<chunk_count, kThreads, 0, stream>>>(partials, found_issue, d, chunks);
    if (cudaGetLastError() != cudaSuccess) return false;
    activity_norm_finalize_kernel<<<1, 1, 0, stream>>>(total, partials, coefficient,
        found_issue, d.contribution, chunk_count, max_norm);
    if (cudaGetLastError() != cudaSuccess) return false;
    activity_clip_kernel<<<chunk_count, kThreads, 0, stream>>>(d, chunks, coefficient, found_issue);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_initialize_moments(
    NsosActivityAwareOptimizerDesc d, unsigned char* initialized) {
    if (!valid_desc(d) || !d.m || !d.v || !initialized) return false;
    activity_initialize_moments_kernel<<<d.n_tensors, kThreads, 0, nsos::gpu::current_stream()>>>(d, initialized);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_adamw_update_deterministic(
    NsosActivityAwareOptimizerDesc d, const NsosMultiTensorChunk* chunks, int chunk_count,
    float beta1, float beta2, float bc1, float bc2, float eps,
    float weight_decay, int* found_nonfinite) {
    if (!valid_desc(d) || !d.w || !d.g || !d.m || !d.v || !d.wd_flags ||
        !d.learning_rates || !found_nonfinite || !valid_chunks(chunks, chunk_count) ||
        d.contribution.abort_issue == found_nonfinite ||
        !valid_adam(beta1, beta2, bc1, bc2, eps, weight_decay)) return false;
    const auto meta = activity_meta(d);
    if (chunk_count > 0) {
        multi_tensor_adamw_update_deterministic_chunked_kernel<<<chunk_count, kThreads, 0, nsos::gpu::current_stream()>>>(
            meta, chunks, chunk_count, beta1, beta2, bc1, bc2, eps, weight_decay, found_nonfinite);
    } else {
        multi_tensor_adamw_update_deterministic_kernel<<<blocks_for_total(d.total), kThreads, 0, nsos::gpu::current_stream()>>>(
            meta, beta1, beta2, bc1, bc2, eps, weight_decay, found_nonfinite);
    }
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool launch_activity_multi_tensor_adamw(NsosActivityAwareOptimizerDesc d,
    const float* gradient_sq_sum, float accumulation_scale, float max_grad_norm,
    float beta1, float beta2, float bc1, float bc2, float lr, float eps,
    float weight_decay, int* found_nonfinite) {
    if (!valid_desc(d) || !d.w || !d.g || !d.m || !d.v || !d.wd_flags ||
        !gradient_sq_sum || !found_nonfinite || d.contribution.abort_issue == found_nonfinite ||
        !std::isfinite(accumulation_scale) || accumulation_scale <= 0 ||
        !std::isfinite(max_grad_norm) || max_grad_norm <= 0 || !std::isfinite(lr) || lr < 0 ||
        !valid_adam(beta1, beta2, bc1, bc2, eps, weight_decay)) return false;
    multi_tensor_adamw_kernel<<<blocks_for_total(d.total), kThreads, 0, nsos::gpu::current_stream()>>>(
        activity_meta(d), gradient_sq_sum, accumulation_scale, max_grad_norm,
        beta1, beta2, bc1, bc2, lr, eps, weight_decay, found_nonfinite);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
__global__ void add_row_broadcast_kernel(float* __restrict__ grad,
                                         const float* __restrict__ row_add,
                                         int rows, int cols) {
    const long long total = static_cast<long long>(rows) * cols;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        grad[i] += row_add[i / cols];
    }
}
}  // namespace

// (auditoria #6) MoE aux-reg device-side: grad do gate [E, d] += imbalance[E]
// sem o round-trip D2H/H2D por step que existia no caminho host.
extern "C" void launch_add_row_broadcast(float* grad, const float* row_add,
                                         int rows, int cols) {
    const long long total = static_cast<long long>(rows) * cols;
    if (total <= 0) return;
    const long long b = nsos::gpu::ceil_div_positive(
        total, static_cast<long long>(kThreads));
    add_row_broadcast_kernel<<<static_cast<int>(b < kMaxBlocks ? b : kMaxBlocks),
                               kThreads>>>(grad, row_add, rows, cols);
}
