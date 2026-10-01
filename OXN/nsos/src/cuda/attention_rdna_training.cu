// FA-inspired bounded scheduler; native RDNA3 QK tiles, not a Blackwell port.
// Hybrid on purpose: FP32 PV/VJP avoids lowp P/dS rounding and its extra STE.
#define NSOS_INCLUDE_ROCWMMA 1
#include "gpu_backend.h"
#include "cuda/attention_rdna_training.cuh"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>

#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "RDNA attention v1 requires finite checks and precise expf; compile this source with -fno-fast-math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && \
    defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "RDNA attention v1 must be compiled for wave32"
#endif
#endif

namespace nsos::attention_rdna {
namespace {
struct Range { const void* p; std::size_t bytes; };
bool separated(std::initializer_list<Range> reads, std::initializer_list<Range> writes) {
    auto good = [](Range a) {
        const auto start = reinterpret_cast<std::uintptr_t>(a.p);
        return a.p && start % 4 == 0 && a.bytes &&
            start <= std::numeric_limits<std::uintptr_t>::max() - a.bytes;
    };
    auto overlaps = [](Range a, Range b) {
        const auto x = reinterpret_cast<std::uintptr_t>(a.p);
        const auto y = reinterpret_cast<std::uintptr_t>(b.p);
        return x < y + b.bytes && y < x + a.bytes;
    };
    for (auto a : reads) if (!good(a)) return false;
    for (auto a : writes) if (!good(a)) return false;
    for (auto a = writes.begin(); a != writes.end(); ++a) {
        for (auto b = a + 1; b != writes.end(); ++b) if (overlaps(*a, *b)) return false;
        for (auto b : reads) if (overlaps(*a, b)) return false;
    }
    return true;
}

#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
constexpr int Tile = 16;
__device__ std::size_t idx(int b, int token, int h, int heads, Shape s) {
    return ((static_cast<std::size_t>(b) * s.sequence + token) * heads + h) * s.head_dim;
}
__device__ std::size_t row(int b, int token, int h, Shape s) {
    return (static_cast<std::size_t>(b) * s.sequence + token) * s.query_heads + h;
}
__device__ bool visible(int i, int j, int n, Shape s) {
    return i < n && j < n && j <= i && i - j < s.window;
}

// One owner per batch: fixed reduction of lane flags, no atomic status writes.
template<bool Backward>
__global__ __launch_bounds__(32) void preflight(Shape s, const float* q,
    const float* k, const float* v, const float* go, const int* valid, int* status) {
    __shared__ int flags[32];
    const int b = blockIdx.x, t = threadIdx.x, n = valid[b];
    if constexpr (Backward) { if (status[b] != 0) return; }
    if (n < 0 || n > s.sequence) {
        if (t == 0) status[b] = static_cast<int>(DeviceStatus::InvalidPrefix);
        return;
    }
    int bad = 0;
    const std::size_t nq = static_cast<std::size_t>(n) * s.query_heads * s.head_dim;
    const std::size_t nk = static_cast<std::size_t>(n) * s.kv_heads * s.head_dim;
    const auto qb = static_cast<std::size_t>(b) * s.sequence * s.query_heads * s.head_dim;
    const auto kb = static_cast<std::size_t>(b) * s.sequence * s.kv_heads * s.head_dim;
    for (std::size_t x = t; x < nq; x += 32) {
        const float a = Backward ? go[qb + x] : q[qb + x];
        if (!isfinite(a) || fabsf(a) > 64.0f) bad = 1;
    }
    if constexpr (!Backward) {
        for (std::size_t x = t; x < nk; x += 32)
            if (!isfinite(k[kb + x]) || fabsf(k[kb + x]) > 64.0f ||
                !isfinite(v[kb + x]) || fabsf(v[kb + x]) > 64.0f) bad = 1;
    }
    flags[t] = bad;
    __syncthreads();
    if (t == 0) {
        int any = 0;
        for (int x = 0; x < 32; ++x) any |= flags[x];
        status[b] = any ? static_cast<int>(DeviceStatus::InvalidOperand) : 0;
    }
}

// Matrix B is column-major: key rows are its columns, avoiding a guessed
// fragment layout or LDS transpose. Every lane participates, including tails.
template<class Low, int Pitch>
__device__ void qk_tile(const Low* qt, const Low* kt, float* score, int d) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, Low, rocwmma::row_major> a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, Low, rocwmma::col_major> b;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c;
    rocwmma::fill_fragment(c, 0.0f);
    for (int x = 0; x < d; x += 16) {
        rocwmma::load_matrix_sync(a, qt + x, Pitch);
        rocwmma::load_matrix_sync(b, kt + x, Pitch);
        rocwmma::mma_sync(c, a, b, c);
    }
    rocwmma::store_matrix_sync(score, c, Tile, rocwmma::layout_t::mem_row_major);
    __syncthreads();
#else
    // Runtime selection rejects non-gfx11 code. Never let a wrong code object
    // silently produce a successful no-op if that selection contract breaks.
    __builtin_trap();
#endif
}
template<class Low, int Pitch>
__device__ void load_tile(Low* tile, const float* src, int b, int h,
    int base, int heads, int n, Shape s) {
    for (int x = threadIdx.x; x < Tile * Pitch; x += 32) {
        const int token = base + x / Pitch, d = x % Pitch;
        tile[x] = static_cast<Low>(token < n && d < s.head_dim ?
            src[idx(b, token, h, heads, s) + d] : 0.0f);
    }
    __syncthreads();
}

template<class Low, int Pitch>
__global__ __launch_bounds__(32) void fwd(Shape s, const float* q, const float* k,
    const float* v, const int* valid, const int* status, float* out, float* mx, float* inv) {
    __shared__ __align__(32) Low qt[Tile * Pitch], kt[Tile * Pitch], vt[Tile * Pitch];
    __shared__ __align__(32) float scores[Tile * Tile], p[Tile * Tile], accum[Tile * Pitch];
    __shared__ float m[Tile], l[Tile], alpha[Tile];
    const int b = blockIdx.z, h = blockIdx.y, qb = blockIdx.x * Tile;
    const int kh = h / (s.query_heads / s.kv_heads), t = threadIdx.x;
    const int n = status[b] == 0 ? valid[b] : 0;
    if (qb >= n) {
        for (int x = t; x < Tile * s.head_dim; x += 32)
            if (qb + x / s.head_dim < s.sequence)
                out[idx(b, qb + x / s.head_dim, h, s.query_heads, s) + x % s.head_dim] = 0;
        if (t < Tile && qb + t < s.sequence) { mx[row(b,qb+t,h,s)] = 0; inv[row(b,qb+t,h,s)] = 0; }
        return;
    }
    load_tile<Low, Pitch>(qt, q, b, h, qb, s.query_heads, n, s);
    for (int x = t; x < Tile * Pitch; x += 32) accum[x] = 0;
    if (t < Tile) { m[t] = -INFINITY; l[t] = 0; }
    __syncthreads();
    const int first = (max(0, qb - min(s.window, s.sequence) + 1) / Tile) * Tile;
    const int last = min(n, qb + Tile);
    for (int kb = first; kb < last; kb += Tile) {
        load_tile<Low, Pitch>(kt, k, b, kh, kb, s.kv_heads, n, s);
        load_tile<Low, Pitch>(vt, v, b, kh, kb, s.kv_heads, n, s);
        qk_tile<Low, Pitch>(qt, kt, scores, s.head_dim);
        if (t < Tile) {
            float next = m[t];
            for (int j = 0; j < Tile; ++j)
                if (visible(qb+t,kb+j,n,s)) next = fmaxf(next, scores[t*Tile+j] * s.scale);
            // An empty masked tile never evaluates -inf - -inf.
            alpha[t] = l[t] == 0 ? 0 : expf(m[t] - next);
            float sum = 0;
            for (int j = 0; j < Tile; ++j) {
                const float e = visible(qb+t,kb+j,n,s) ? expf(scores[t*Tile+j]*s.scale-next) : 0;
                p[t*Tile+j] = e; sum += e;
            }
            l[t] = alpha[t]*l[t] + sum; m[t] = next;
        }
        __syncthreads();
        for (int x = t; x < Tile * Pitch; x += 32) {
            const int i = x / Pitch, d = x % Pitch;
            float value = accum[x] * alpha[i];
            for (int j = 0; j < Tile; ++j) value += p[i*Tile+j] * static_cast<float>(vt[j*Pitch+d]);
            accum[x] = value;
        }
        __syncthreads();
    }
    for (int x = t; x < Tile * Pitch; x += 32) {
        const int i = qb + x / Pitch, d = x % Pitch;
        if (i < s.sequence && d < s.head_dim)
            out[idx(b,i,h,s.query_heads,s)+d] = i < n ? accum[x] / l[x/Pitch] : 0;
    }
    if (t < Tile && qb+t < s.sequence) {
        mx[row(b,qb+t,h,s)] = qb+t < n ? m[t] : 0;
        inv[row(b,qb+t,h,s)] = qb+t < n ? 1.0f/l[t] : 0;
    }
}

template<class Low, int Pitch>
__global__ __launch_bounds__(32) void bwd_q(Shape s, const float* q, const float* k,
    const float* v, const float* out, const float* go, const int* valid, const int* status,
    const float* mx, const float* inv, float* delta, float* dq) {
    __shared__ __align__(32) Low qt[Tile * Pitch], kt[Tile * Pitch];
    __shared__ __align__(32) float scores[Tile * Tile], ds[Tile * Tile], acc[Tile * Pitch];
    __shared__ float dot[Tile];
    const int b = blockIdx.z, h = blockIdx.y, qb = blockIdx.x * Tile, t = threadIdx.x;
    const int kh = h / (s.query_heads / s.kv_heads);
    const int n = status[b] == 0 ? valid[b] : 0;
    if (t < Tile) {
        float value = 0;
        if (qb+t < n) for (int d = 0; d < s.head_dim; ++d)
            value += go[idx(b,qb+t,h,s.query_heads,s)+d] * out[idx(b,qb+t,h,s.query_heads,s)+d];
        dot[t] = value;
        if (qb+t < s.sequence) delta[row(b,qb+t,h,s)] = value;
    }
    for (int x = t; x < Tile * Pitch; x += 32) acc[x] = 0;
    __syncthreads();
    if (qb < n) {
        load_tile<Low, Pitch>(qt, q, b, h, qb, s.query_heads, n, s);
        const int first = (max(0,qb-min(s.window,s.sequence)+1) / Tile) * Tile;
        for (int kb = first; kb < min(n,qb+Tile); kb += Tile) {
            load_tile<Low, Pitch>(kt, k, b, kh, kb, s.kv_heads, n, s);
            qk_tile<Low, Pitch>(qt, kt, scores, s.head_dim);
            for (int x = t; x < Tile * Tile; x += 32) {
                const int i = qb+x/Tile, j = kb+x%Tile;
                float value = 0;
                if (visible(i,j,n,s)) {
                    float dp = 0;
                    for (int d = 0; d < s.head_dim; ++d)
                        dp += go[idx(b,i,h,s.query_heads,s)+d] *
                            static_cast<float>(static_cast<Low>(v[idx(b,j,kh,s.kv_heads,s)+d]));
                    const auto r = row(b,i,h,s);
                    const float p = expf(scores[x]*s.scale-mx[r]) * inv[r];
                    value = p * (dp-dot[x/Tile]) * s.scale;
                }
                ds[x] = value;
            }
            __syncthreads();
            for (int x = t; x < Tile * Pitch; x += 32) {
                const int i = x/Pitch, d = x%Pitch;
                float value = acc[x];
                for (int j = 0; j < Tile; ++j) value += ds[i*Tile+j] * static_cast<float>(kt[j*Pitch+d]);
                acc[x] = value;
            }
            __syncthreads();
        }
    }
    for (int x = t; x < Tile * Pitch; x += 32) {
        const int i = qb+x/Pitch, d = x%Pitch;
        if (i < s.sequence && d < s.head_dim) dq[idx(b,i,h,s.query_heads,s)+d] = acc[x];
    }
}

// A CTA owns one KV tile across ALL its GQA heads/query tiles. Fixed head,
// tile, row order; no atomics and no B*H*S*D partial gradient allocation.
template<class Low, int Pitch>
__global__ __launch_bounds__(32) void bwd_kv(Shape s, const float* q, const float* k,
    const float* v, const float* go, const int* valid, const int* status,
    const float* mx, const float* inv, const float* delta, float* dk, float* dv) {
    __shared__ __align__(32) Low qt[Tile * Pitch], kt[Tile * Pitch];
    __shared__ __align__(32) float scores[Tile*Tile], p[Tile*Tile], ds[Tile*Tile];
    __shared__ float ak[Tile*Pitch], av[Tile*Pitch];
    const int b = blockIdx.z, kh = blockIdx.y, kb = blockIdx.x * Tile, t = threadIdx.x;
    const int n = status[b] == 0 ? valid[b] : 0, group = s.query_heads / s.kv_heads;
    for (int x = t; x < Tile*Pitch; x += 32) { ak[x] = 0; av[x] = 0; }
    __syncthreads();
    if (kb < n) {
        load_tile<Low, Pitch>(kt,k,b,kh,kb,s.kv_heads,n,s);
        const int end = min(n, kb+Tile-1+min(s.window,s.sequence));
        for (int h = kh*group; h < (kh+1)*group; ++h) {
            for (int qb = kb; qb < end; qb += Tile) {
                load_tile<Low, Pitch>(qt,q,b,h,qb,s.query_heads,n,s);
                qk_tile<Low, Pitch>(qt,kt,scores,s.head_dim);
                for (int x = t; x < Tile*Tile; x += 32) {
                    const int i = qb+x/Tile, j = kb+x%Tile;
                    float prob = 0, value = 0;
                    if (visible(i,j,n,s)) {
                        float dp = 0;
                        for (int d = 0; d < s.head_dim; ++d)
                            dp += go[idx(b,i,h,s.query_heads,s)+d] *
                                static_cast<float>(static_cast<Low>(v[idx(b,j,kh,s.kv_heads,s)+d]));
                        const auto r = row(b,i,h,s);
                        prob = expf(scores[x]*s.scale-mx[r]) * inv[r];
                        value = prob * (dp-delta[r]) * s.scale;
                    }
                    p[x] = prob; ds[x] = value;
                }
                __syncthreads();
                for (int x = t; x < Tile*Pitch; x += 32) {
                    const int j = x/Pitch, d = x%Pitch;
                    float gk = ak[x], gv = av[x];
                    for (int i = 0; i < Tile; ++i) {
                        gk += ds[i*Tile+j] * static_cast<float>(qt[i*Pitch+d]);
                        // Do not read padded dO: 0*NaN is not zero.
                        if (visible(qb+i,kb+j,n,s) && d < s.head_dim)
                            gv += p[i*Tile+j] * go[idx(b,qb+i,h,s.query_heads,s)+d];
                    }
                    ak[x] = gk; av[x] = gv;
                }
                __syncthreads();
            }
        }
    }
    for (int x = t; x < Tile*Pitch; x += 32) {
        const int j = kb+x/Pitch, d = x%Pitch;
        if (j < s.sequence && d < s.head_dim) {
            const auto dst = idx(b,j,kh,s.kv_heads,s)+d;
            dk[dst] = ak[x]; dv[dst] = av[x];
        }
    }
}

template<class Low, int Pitch> bool attributes_ok(std::size_t max_shared) {
    static_assert(sizeof(Low) == 2, "attention lowp LDS accounting requires 16-bit operands");
    hipFuncAttributes a{}, b{}, c{};
    return hipFuncGetAttributes(&a, reinterpret_cast<const void*>(fwd<Low,Pitch>)) == hipSuccess &&
        hipFuncGetAttributes(&b, reinterpret_cast<const void*>(bwd_q<Low,Pitch>)) == hipSuccess &&
        hipFuncGetAttributes(&c, reinterpret_cast<const void*>(bwd_kv<Low,Pitch>)) == hipSuccess &&
        a.maxThreadsPerBlock >= 32 && b.maxThreadsPerBlock >= 32 && c.maxThreadsPerBlock >= 32 &&
        a.sharedSizeBytes <= max_shared && b.sharedSizeBytes <= max_shared && c.sharedSizeBytes <= max_shared;
}
template<class Low> bool attributes_for(int d, std::size_t max_shared) {
    if (d <= 16) return attributes_ok<Low,16>(max_shared);
    if (d <= 32) return attributes_ok<Low,32>(max_shared);
    if (d <= 64) return attributes_ok<Low,64>(max_shared);
    if (d <= 128) return attributes_ok<Low,128>(max_shared);
    return attributes_ok<Low,256>(max_shared);
}
struct Args {
    Shape s;
    const float *q, *k, *v, *out, *go, *mx, *inv;
    const int* valid;
    int* status;
    float *o, *m, *l, *delta, *dq, *dk, *dv;
};
template<bool Backward, class Low, int Pitch> bool launch(Args a) {
    const auto stream = gpu::current_stream();
    const dim3 grid((a.s.sequence+Tile-1)/Tile,a.s.query_heads,a.s.batch);
    if constexpr (!Backward) {
        fwd<Low,Pitch><<<grid,32,0,stream>>>(a.s,a.q,a.k,a.v,a.valid,a.status,a.o,a.m,a.l);
        return hipGetLastError() == hipSuccess;
    } else {
        bwd_q<Low,Pitch><<<grid,32,0,stream>>>(a.s,a.q,a.k,a.v,a.out,a.go,a.valid,a.status,a.mx,a.inv,a.delta,a.dq);
        if (hipGetLastError() != hipSuccess) return false;
        const dim3 kvgrid(grid.x,a.s.kv_heads,a.s.batch);
        bwd_kv<Low,Pitch><<<kvgrid,32,0,stream>>>(a.s,a.q,a.k,a.v,a.go,a.valid,a.status,a.mx,a.inv,a.delta,a.dk,a.dv);
        return hipGetLastError() == hipSuccess;
    }
}
template<bool Backward, class Low> bool launch_for(Args a) {
    if (a.s.head_dim <= 16) return launch<Backward,Low,16>(a);
    if (a.s.head_dim <= 32) return launch<Backward,Low,32>(a);
    if (a.s.head_dim <= 64) return launch<Backward,Low,64>(a);
    if (a.s.head_dim <= 128) return launch<Backward,Low,128>(a);
    return launch<Backward,Low,256>(a);
}
template<bool Backward> bool dispatch(Args a) {
    preflight<Backward><<<a.s.batch,32,0,gpu::current_stream()>>>(a.s,a.q,a.k,a.v,a.go,a.valid,a.status);
    if (hipGetLastError() != hipSuccess) return false;
    return a.s.precision == Precision::BF16 ? launch_for<Backward,rocwmma::bfloat16_t>(a) :
        launch_for<Backward,rocwmma::float16_t>(a);
}
#endif
} // namespace

bool supported(const Shape& s) {
    if (!geometry_eligible(s)) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    int device = -1;
    hipDeviceProp_t p{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&p,device) != hipSuccess) return false;
    const std::string arch = std::string(p.gcnArchName).substr(0,std::string(p.gcnArchName).find(':'));
    if (p.warpSize != 32 || (arch != "gfx1100" && arch != "gfx1101" && arch != "gfx1102") ||
        p.sharedMemPerBlock < 52352 || p.maxThreadsPerBlock < 32 ||
        p.maxGridSize[0] < (s.sequence+15)/16 || p.maxGridSize[1] < s.query_heads ||
        p.maxGridSize[2] < s.batch || p.maxGridSize[0] < s.batch) return false;
    bool compiled = false;
    for (const auto& info : gpu::enumerate_devices()) if (info.index == device) compiled = info.compiled;
    if (!compiled) return false;
    hipFuncAttributes pf{}, pb{};
    if (hipFuncGetAttributes(&pf,reinterpret_cast<const void*>(preflight<false>)) != hipSuccess ||
        hipFuncGetAttributes(&pb,reinterpret_cast<const void*>(preflight<true>)) != hipSuccess ||
        pf.maxThreadsPerBlock < 32 || pb.maxThreadsPerBlock < 32 ||
        pf.sharedSizeBytes > p.sharedMemPerBlock || pb.sharedSizeBytes > p.sharedMemPerBlock) return false;
    return s.precision == Precision::BF16 ? attributes_for<rocwmma::bfloat16_t>(s.head_dim,p.sharedMemPerBlock) :
        attributes_for<rocwmma::float16_t>(s.head_dim,p.sharedMemPerBlock);
#else
    return false;
#endif
}

bool forward(const Shape& s, const float* q, const float* k, const float* v,
    const int* valid, float* out, float* mx, float* inv, int* status) {
    if (!geometry_eligible(s)) return false;
    const auto nq = query_elements(s)*sizeof(float), nk = kv_elements(s)*sizeof(float);
    const auto nr = row_elements(s)*sizeof(float), nb = static_cast<std::size_t>(s.batch)*sizeof(int);
    if (!separated({{q,nq},{k,nk},{v,nk},{valid,nb}},
        {{out,nq},{mx,nr},{inv,nr},{status,nb}}) || !supported(s)) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    return dispatch<false>({s,q,k,v,nullptr,nullptr,nullptr,nullptr,valid,status,out,mx,inv,nullptr,nullptr,nullptr,nullptr});
#else
    return false;
#endif
}
bool backward(const Shape& s, const float* q, const float* k, const float* v,
    const float* out, const float* go, const int* valid, const float* mx, const float* inv,
    int* status, float* delta, float* dq, float* dk, float* dv) {
    if (!geometry_eligible(s)) return false;
    const auto nq = query_elements(s)*sizeof(float), nk = kv_elements(s)*sizeof(float);
    const auto nr = row_elements(s)*sizeof(float), nb = static_cast<std::size_t>(s.batch)*sizeof(int);
    if (!separated({{q,nq},{k,nk},{v,nk},{out,nq},{go,nq},{valid,nb},{mx,nr},{inv,nr}},
        {{status,nb},{delta,nr},{dq,nq},{dk,nk},{dv,nk}}) || !supported(s)) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    return dispatch<true>({s,q,k,v,out,go,mx,inv,valid,status,nullptr,nullptr,nullptr,delta,dq,dk,dv});
#else
    return false;
#endif
}
} // namespace nsos::attention_rdna

// ATTENTION OWNED TRAINING INTEGRATION v1
// Integration helpers: no independent numerical policy or CPU fallback.
namespace nsos::attention_rdna {
#ifdef USE_CUDA
namespace {
__device__ float rotate_component(const float* x, std::size_t base, int d,
    int hd, float c, float s, bool inverse) {
    const int half = hd / 2;
    if (d >= 2 * half) return x[base+d];
    if (d < half) return x[base+d]*c + x[base+d+half]*(inverse?s:-s);
    return x[base+d]*c + x[base+d-half]*(inverse?-s:s);
}
__global__ void prepare_kernel(Shape s, int start, const float* q,
    const float* kv, const float* cs, const float* sn, const int* valid,
    float* qr, float* kr, float* v) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int qdim=s.query_heads*s.head_dim, kdim=s.kv_heads*s.head_dim;
    const std::size_t tokens=static_cast<std::size_t>(s.batch)*s.sequence;
    if (x >= tokens*qdim) return;
    const auto token=x/qdim; const int channel=x%qdim, d=channel%s.head_dim;
    const bool active=static_cast<int>(token%s.sequence)<valid[token/s.sequence];
    const int half=s.head_dim/2;
    const auto freq=(token%s.sequence+start)*half+(half?d%half:0);
    // Branch before reading ANY padded payload or table value.
    const float c=active&&d<2*half?cs[freq]:1;
    const float z=active&&d<2*half?sn[freq]:0;
    qr[x]=active?rotate_component(q,x-d,d,s.head_dim,c,z,false):0;
    if (channel<kdim) {
        const auto ki=token*kdim+channel, src=token*(2*kdim)+channel;
        kr[ki]=active?rotate_component(kv,src-d,d,s.head_dim,c,z,false):0;
        v[ki]=active?kv[src+kdim]:0;
    }
}
__global__ void join_kernel(Shape s,int start,const float* dq,const float* dk,
    const float* dv,const float* cs,const float* sn,const int* valid,const int* status,float* q,float* kv) {
    const auto x=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int qdim=s.query_heads*s.head_dim,kdim=s.kv_heads*s.head_dim;
    if (x>=static_cast<std::size_t>(s.batch)*s.sequence*qdim) return;
    const auto token=x/qdim; const int ch=x%qdim,d=ch%s.head_dim,half=s.head_dim/2;
    if(static_cast<int>(token%s.sequence)>=valid[token/s.sequence]||status[token/s.sequence]!=0) {
        q[x]=0;if(ch<kdim){const auto dst=token*(2*kdim)+ch;kv[dst]=0;kv[dst+kdim]=0;}return;
    }
    const auto f=(token%s.sequence+start)*half+(half?d%half:0);
    const float c=d<2*half?cs[f]:1,z=d<2*half?sn[f]:0;
    q[x]=rotate_component(dq,x-d,d,s.head_dim,c,z,true);
    if (ch<kdim) {
        const auto src=token*kdim+ch,dst=token*(2*kdim)+ch;
        kv[dst]=rotate_component(dk,src-d,d,s.head_dim,c,z,true);
        kv[dst+kdim]=dv[src];
    }
}
__global__ void mask_kernel(Shape s,int f,const int* valid,float* x) {
    const auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=static_cast<std::size_t>(s.batch)*s.sequence*f) return;
    const auto token=i/f;
    if (static_cast<int>(token%s.sequence)>=valid[token/s.sequence]) x[i]=0;
}
__global__ void inspect_kernel(Shape s,int f,const int* valid,const float* x,int* status) {
    __shared__ int bad[32];
    const int b=blockIdx.x,t=threadIdx.x;
    int issue=status[b]!=0;
    const int n=valid[b];
    if(n<0||n>s.sequence) issue=1;
    else for(std::size_t i=t;i<static_cast<std::size_t>(n)*f;i+=32)
        if(!isfinite(x[static_cast<std::size_t>(b)*s.sequence*f+i])) issue=1;
    bad[t]=issue; __syncthreads();
    if(t==0) {
        int any=0; for(int i=0;i<32;++i) any|=bad[i];
        if(any&&status[b]==0) status[b]=static_cast<int>(DeviceStatus::NonFiniteResult);
    }
}
__global__ void merge_kernel(const int* status,int n,int* flag) {
    if(threadIdx.x==0) {
        int result=*flag; for(int i=0;i<n;++i) result|=status[i]!=0;
        *flag=result;
    }
}
bool launch_ok() {return cudaGetLastError()==cudaSuccess;}
bool table_ok(const Shape& s,int start,int rows) {
    return geometry_eligible(s)&&query_elements(s)<=static_cast<std::size_t>(std::numeric_limits<int>::max()/2)&&
        start>=0&&rows>=s.sequence&&start<=rows-s.sequence;
}
unsigned blocks(std::size_t n) {return static_cast<unsigned>((n+255)/256);}
}
#endif
bool prepare_projected(const Shape& s,int start,int rows,const float* q,const float* kv,
    const float* cs,const float* sn,const int* valid,float* qr,float* kr,float* v) {
#ifdef USE_CUDA
    if(!table_ok(s,start,rows)||!q||!kv||!cs||!sn||!valid||!qr||!kr||!v) return false;
    prepare_kernel<<<blocks(query_elements(s)),256,0,gpu::current_stream()>>>(s,start,q,kv,cs,sn,valid,qr,kr,v);
    return launch_ok();
#else
    return false;
#endif
}
bool join_projected_gradient(const Shape& s,int start,int rows,const float* dq,const float* dk,
    const float* dv,const float* cs,const float* sn,const int* valid,const int* status,float* q,float* kv) {
#ifdef USE_CUDA
    if(!table_ok(s,start,rows)||!dq||!dk||!dv||!cs||!sn||!valid||!status||!q||!kv) return false;
    join_kernel<<<blocks(query_elements(s)),256,0,gpu::current_stream()>>>(s,start,dq,dk,dv,cs,sn,valid,status,q,kv);
    return launch_ok();
#else
    return false;
#endif
}
bool mask_prefix(const Shape& s,int features,const int* valid,float* x) {
#ifdef USE_CUDA
    if(!geometry_eligible(s)||features<=0||!valid||!x) return false;
    if(static_cast<std::uint64_t>(s.batch)*s.sequence*features>std::numeric_limits<int>::max()) return false;
    mask_kernel<<<blocks(static_cast<std::size_t>(s.batch)*s.sequence*features),256,0,gpu::current_stream()>>>(s,features,valid,x);
    return launch_ok();
#else
    return false;
#endif
}
bool inspect_result(const Shape& s,int features,const int* valid,const float* x,int* status) {
#ifdef USE_CUDA
    if(!geometry_eligible(s)||features<=0||!valid||!x||!status) return false;
    if(static_cast<std::uint64_t>(s.batch)*s.sequence*features>std::numeric_limits<int>::max()) return false;
    inspect_kernel<<<s.batch,32,0,gpu::current_stream()>>>(s,features,valid,x,status);
    return launch_ok();
#else
    return false;
#endif
}
bool merge_device_status(const int* status,int count,int* issue) {
#ifdef USE_CUDA
    if(!status||!issue||status==issue||count<=0) return false;
    merge_kernel<<<1,1,0,gpu::current_stream()>>>(status,count,issue);
    return launch_ok();
#else
    return false;
#endif
}
}
