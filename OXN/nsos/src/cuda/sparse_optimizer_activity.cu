#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/sparse_optimizer_activity.cuh"

namespace {
__device__ bool active(NsosActivityAwareOptimizerDesc d, int t) {
    if (d.contribution.abort_issue && *d.contribution.abort_issue) return false;
    return !d.contribution.predicates || !d.contribution.predicates[t] || *d.contribution.predicates[t] != 0;
}
__device__ bool contributed(NsosActivityAwareOptimizerDesc d, int t) {
    return !d.contribution.predicates || !d.contribution.predicates[t] || *d.contribution.predicates[t] != 0;
}
__global__ void merge_kernel(int* issue, const int* const* sources, int count) {
    if (threadIdx.x == 0) for (int i = 0; i < count; ++i)
        if (!sources[i] || *sources[i]) *issue = 1;
}
__global__ void version_preflight_kernel(NsosActivityAwareOptimizerDesc d, const uint64_t* versions, int* issue) {
    const int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<d.n_tensors && active(d,t) && versions[t]==UINT64_MAX)atomicExch(issue,1);
}
__global__ void publish_kernel(NsosActivityAwareOptimizerDesc d, uint64_t* versions,
    unsigned char* cache_valid, const int* output_issue) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < d.n_tensors && !*output_issue && active(d, t)) {
        ++versions[t]; if (cache_valid) cache_valid[t] = 0;
    }
}
__global__ void snapshot_kernel(NsosActivityAwareOptimizerDesc d,
    const unsigned char* initialized, unsigned char* backup_initialized, float* scratch) {
    const int t = blockIdx.x;
    if (threadIdx.x == 0) backup_initialized[t] = initialized[t];
    // Save candidates even when preflight aborts, but never touch inactive
    // storage. Host invokes restore only after a successful pre-update gate.
    if (!contributed(d, t)) return;
    const auto begin = d.offsets[t], end = d.offsets[t+1];
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const auto j = i - begin;
        scratch[i] = d.w[t][j]; scratch[d.total + i] = d.g[t][j];
        if (initialized[t]) {
            scratch[2*d.total+i] = d.m[t][j]; scratch[3*d.total+i] = d.v[t][j];
        }
    }
}
__global__ void restore_kernel(NsosActivityAwareOptimizerDesc d,
    unsigned char* initialized, const unsigned char* backup_initialized, const float* scratch) {
    const int t = blockIdx.x;
    if (contributed(d, t)) {
        const auto begin = d.offsets[t], end = d.offsets[t+1];
        for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
            const auto j = i - begin;
            d.w[t][j] = scratch[i]; d.g[t][j] = scratch[d.total+i];
            if (backup_initialized[t]) {
                d.m[t][j] = scratch[2*d.total+i]; d.v[t][j] = scratch[3*d.total+i];
            }
        }
    }
    if (threadIdx.x == 0) initialized[t] = backup_initialized[t];
}
bool valid(NsosActivityAwareOptimizerDesc d) {
    return d.n_tensors > 0 && d.total > 0 && d.w && d.g && d.m && d.v && d.offsets;
}
}
extern "C" bool launch_activity_merge_abort(int* issue, const int* const* sources, int count) {
    if (!issue || count < 0 || (count && !sources)) return false;
    merge_kernel<<<1, 1, 0, nsos::gpu::current_stream()>>>(issue, sources, count);
    return cudaGetLastError() == cudaSuccess;
}
extern "C" bool launch_activity_preflight_versions(NsosActivityAwareOptimizerDesc d, const uint64_t* versions, int* issue) {
    if(!valid(d)||!versions||!issue)return false;
    version_preflight_kernel<<<(d.n_tensors+255)/256,256,0,nsos::gpu::current_stream()>>>(d,versions,issue);
    return cudaGetLastError()==cudaSuccess;
}
extern "C" bool launch_activity_publish_versions(NsosActivityAwareOptimizerDesc d,
    uint64_t* versions, unsigned char* cache_valid, const int* output_issue) {
    if (!valid(d) || !versions || !output_issue) return false;
    publish_kernel<<<(d.n_tensors+255)/256, 256, 0, nsos::gpu::current_stream()>>>(d, versions, cache_valid, output_issue);
    return cudaGetLastError() == cudaSuccess;
}
extern "C" bool launch_activity_snapshot(NsosActivityAwareOptimizerDesc d,
    const unsigned char* initialized, unsigned char* backup_initialized, float* scratch) {
    if (!valid(d) || !initialized || !backup_initialized || !scratch) return false;
    snapshot_kernel<<<d.n_tensors, 256, 0, nsos::gpu::current_stream()>>>(d, initialized, backup_initialized, scratch);
    return cudaGetLastError() == cudaSuccess;
}
extern "C" bool launch_activity_restore(NsosActivityAwareOptimizerDesc d,
    unsigned char* initialized, const unsigned char* backup_initialized, const float* scratch) {
    if (!valid(d) || !initialized || !backup_initialized || !scratch) return false;
    restore_kernel<<<d.n_tensors, 256, 0, nsos::gpu::current_stream()>>>(d, initialized, backup_initialized, scratch);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
__global__ void activity_criticality_metrics_kernel(
    float* const* w, const unsigned long long* offsets, const int* fan_in,
    int n_tensors, const unsigned char* const* predicates, const int* const* issues, float* gammas, float* gains) {
    const int t = static_cast<int>(blockIdx.x);
    if (t >= n_tensors) return;
    if ((predicates && predicates[t] && !*predicates[t]) || (issues && issues[t] && *issues[t])) {
        if (!threadIdx.x) { gammas[t]=0; gains[t]=0; } return;
    }
    const unsigned long long n64 = offsets[t + 1] - offsets[t];
    if (n64 == 0) {
        if (threadIdx.x == 0) {
            gammas[t] = 0.0f;
            gains[t] = 0.0f;
        }
        return;
    }
    __shared__ float abs_partial[256];
    __shared__ unsigned int zero_partial[256];
    float local_abs = 0.0f;
    for (unsigned long long i = threadIdx.x; i < n64; i += blockDim.x) {
        local_abs += fabsf(w[t][i]);
    }
    abs_partial[threadIdx.x] = local_abs;
    __syncthreads();
    for (int stride = 256 / 2; stride > 0; stride >>= 1) {
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
    for (int stride = 256 / 2; stride > 0; stride >>= 1) {
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


__global__ void activity_criticality_gradient_kernel(float* const* w, float* const* g,
    const unsigned long long* offsets, const float* coefficients, int count,
    const unsigned char* const* predicates, const int* const* issues) {
    const int t=blockIdx.x;
    if(t>=count || (predicates && predicates[t] && !*predicates[t]) ||
        (issues && issues[t] && *issues[t]))return;
    for(auto i=static_cast<unsigned long long>(threadIdx.x);i<offsets[t+1]-offsets[t];i+=blockDim.x) {
        const float value=w[t][i];const float sign=value>0?1.0f:value<0?-1.0f:0.0f;
        g[t][i]+=coefficients[t]*sign;
    }
}
}
extern "C" bool launch_activity_criticality_metrics(float* const* w, const unsigned long long* offsets,
    const int* fan_in,int count,const unsigned char* const* predicates,const int* const* issues,float* gammas,float* gains) {
    if(!w||!offsets||!fan_in||count<=0||!gammas||!gains)return false;
    activity_criticality_metrics_kernel<<<count,256,0,nsos::gpu::current_stream()>>>(w,offsets,fan_in,count,predicates,issues,gammas,gains);
    return cudaGetLastError()==cudaSuccess;
}
extern "C" bool launch_activity_criticality_gradient(float* const* w,float* const* g,const unsigned long long* offsets,
    const float* coefficients,int count,const unsigned char* const* predicates,const int* const* issues) {
    if(!w||!g||!offsets||!coefficients||count<=0)return false;
    activity_criticality_gradient_kernel<<<count,256,0,nsos::gpu::current_stream()>>>(w,g,offsets,coefficients,count,predicates,issues);
    return cudaGetLastError()==cudaSuccess;
}
