// fused_optimizer_kernels.cu — otimizador multi-tensor (item #4 da auditoria).
//
// ANTES: apply_optimizer_step emitia ~3-4 kernels POR PARÂMETRO por step
// (scale 1/acc, norm, [clip-scale], adamw) — ~1850-2450 launches/step para os
// ~614 parâmetros do v11; medido opt=226-490 ms/step na T4.
//
// AGORA (padrão Apex/PyTorch-foreach, variante busca-binária): o passo inteiro
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
// sobem por step num único H2D (~30 KB) porque os ponteiros de grad mudam a
// cada backward (add_grad cria tensor novo).  Buffer device persistente
// cresce sob demanda.  Busca binária por elemento sobre offsets (log2(614)≈10
// passos em L2) — simples e suficiente; chunk-lists ficam como otimização
// futura se o perfil mandar.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

struct FusedOptMeta {
    float* const* w;
    float* const* g;
    float* const* m;
    float* const* v;
    const unsigned long long* offsets;  // [n_tensors + 1] prefixo de elementos
    const unsigned char* wd_flags;      // [n_tensors]
    int n_tensors;
    unsigned long long total;
};

__device__ inline int find_tensor(const unsigned long long* offsets,
                                  int n_tensors,
                                  unsigned long long i) {
    int lo = 0;
    int hi = n_tensors - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
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
                                          float gscale,
                                          float beta1,
                                          float beta2,
                                          float bc1,
                                          float bc2,
                                          float lr,
                                          float eps,
                                          float weight_decay) {
    for (unsigned long long i =
             blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
         i < meta.total;
         i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
        const int t = find_tensor(meta.offsets, meta.n_tensors, i);
        const unsigned long long k = i - meta.offsets[t];
        const float g = meta.g[t][k] * gscale;
        const float m = beta1 * meta.m[t][k] + (1.0f - beta1) * g;
        const float v = beta2 * meta.v[t][k] + (1.0f - beta2) * g * g;
        meta.m[t][k] = m;
        meta.v[t][k] = v;
        float w = meta.w[t][k];
        if (meta.wd_flags[t]) {
            w -= lr * weight_decay * w;
        }
        meta.w[t][k] = w - lr * (m / bc1) / (sqrtf(v / bc2) + eps);
    }
}

inline int blocks_for_total(unsigned long long total) {
    const unsigned long long b = (total + kThreads - 1) / kThreads;
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
    FusedOptMeta meta{w, g, m, v, offsets, wd_flags, n_tensors, total};
    multi_tensor_sqsum_kernel<<<blocks_for_total(total), kThreads>>>(accum, meta);
}

extern "C" void launch_multi_tensor_adamw(float* const* w,
                                          float* const* g,
                                          float* const* m,
                                          float* const* v,
                                          const unsigned long long* offsets,
                                          const unsigned char* wd_flags,
                                          int n_tensors,
                                          unsigned long long total,
                                          float gscale,
                                          float beta1,
                                          float beta2,
                                          float bc1,
                                          float bc2,
                                          float lr,
                                          float eps,
                                          float weight_decay) {
    if (total == 0 || n_tensors <= 0) return;
    FusedOptMeta meta{w, g, m, v, offsets, wd_flags, n_tensors, total};
    multi_tensor_adamw_kernel<<<blocks_for_total(total), kThreads>>>(
        meta, gscale, beta1, beta2, bc1, bc2, lr, eps, weight_decay);
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
    const long long b = (total + kThreads - 1) / kThreads;
    add_row_broadcast_kernel<<<static_cast<int>(b < kMaxBlocks ? b : kMaxBlocks),
                               kThreads>>>(grad, row_add, rows, cols);
}
