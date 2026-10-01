#define NSOS_INCLUDE_ROCWMMA 1
#include "../../include/cuda/kan_kernels.cuh"

#ifdef USE_CUDA

#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/moe_training_wmma.cuh"
#include <climits>
#include <stdexcept>

namespace nsos {
namespace cuda {

namespace {

constexpr int kKanThreads = 256;
constexpr int kTile = 16;
__device__ float kan_operand(float value, int mode) {
    if (mode == 1) {
#if defined(NSOS_GPU_BACKEND_HIP)
        return static_cast<float>(hip_bfloat16(value));
#else
        return __bfloat162float(__float2bfloat16(value));
#endif
    }
    return mode == 2 ? __half2float(__float2half_rn(value)) : value;
}
__device__ float kan_basis(float value, const float* centers, const float* widths, int g) {
    const float width = fmaxf(widths[g], 1e-3f);
    const float diff = (value - centers[g]) / width;
    return expf(-0.5f * diff * diff);
}
__global__ __launch_bounds__(256) void kan_qat_partials(const float* weight, float* partials, int n) {
    __shared__ float tree[256];
    float sum = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
         i < static_cast<size_t>(n); i += static_cast<size_t>(gridDim.x) * 256) sum += fabsf(weight[i]);
    tree[threadIdx.x] = sum;
    __syncthreads();
    for (int stride=128; stride; stride>>=1) {
        if (threadIdx.x < stride) tree[threadIdx.x] += tree[threadIdx.x+stride];
        __syncthreads();
    }
    if (!threadIdx.x) partials[blockIdx.x] = tree[0];
}
__global__ void kan_qat_scale(const float* partials, float* scale, int n, int blocks) {
    float sum = 0;
    for (int b=0;b<blocks;++b) sum += partials[b];
    scale[0] = sum / n + 1e-8f;
}
__global__ __launch_bounds__(256) void kan_qat_quantize(
    const float* weight, float* effective, const float* scale, int n) {
    const float s = scale[0], inv = 1.0f / (s + 1e-8f);
    for (size_t i=static_cast<size_t>(blockIdx.x)*256+threadIdx.x;
         i<static_cast<size_t>(n); i+=static_cast<size_t>(gridDim.x)*256) {
        const float v = weight[i] * inv;
        effective[i] = (v>0.5f?1.0f:v< -0.5f?-1.0f:0.0f) * s;
    }
}
// Kind0: implicit-basis @ W^T; Kind1: dY^T @ implicit-basis.
// LDS padding avoids repeated bank conflicts on the transposed reads.
template<int Kind>
__global__ __launch_bounds__(256) void kan_rbf_gemm(
    const float* input, const float* operand, const float* centers,
    const float* widths, const float* base, const float* bias, float* output,
    int rows, int inputs, int outputs, int grid, int mode) {
    const int features = inputs * grid;
    const int row = blockIdx.y * kTile + threadIdx.y, col = blockIdx.x * kTile + threadIdx.x;
    const int m = Kind == 0 ? rows : outputs, n = Kind == 0 ? outputs : features;
    const int k = Kind == 0 ? features : rows;
    __shared__ float left[kTile][kTile+1], right[kTile][kTile+1];
    float sum = 0;
    for (int start=0;start<k;start+=kTile) {
        const int ak=start+threadIdx.x, bk=start+threadIdx.y;
        float a=0, b=0;
        if constexpr (Kind==0) {
            if (row<rows && ak<features) a=kan_basis(input[static_cast<size_t>(row)*inputs+ak/grid],centers,widths,ak%grid);
            if (col<outputs && bk<features) b=operand[static_cast<size_t>(col)*features+bk];
        } else {
            if (row<outputs && ak<rows) a=operand[static_cast<size_t>(ak)*outputs+row];
            if (col<features && bk<rows) b=kan_basis(input[static_cast<size_t>(bk)*inputs+col/grid],centers,widths,col%grid);
        }
        left[threadIdx.y][threadIdx.x]=kan_operand(a,mode);
        right[threadIdx.y][threadIdx.x]=kan_operand(b,mode);
        __syncthreads();
        #pragma unroll
        for (int t=0;t<kTile;++t) sum=fmaf(left[threadIdx.y][t],right[t][threadIdx.x],sum);
        __syncthreads();
    }
    if (row>=m || col>=n) return;
    const size_t i=static_cast<size_t>(row)*n+col;
    if constexpr (Kind==0) output[i]=(base[i]+sum)+bias[col];
    else output[i]=sum;
}
__global__ __launch_bounds__(256) void kan_rbf_dx(
    const float* input, const float* upstream, const float* weight,
    const float* centers, const float* widths, float* grad_input,
    int rows, int inputs, int outputs, int grid, int mode) {
    const int row=blockIdx.y*kTile+threadIdx.y, feature=blockIdx.x*kTile+threadIdx.x;
    __shared__ float dy[kTile][kTile+1], w[kTile][kTile+1];
    const bool valid=row<rows && feature<inputs;
    const size_t i=static_cast<size_t>(row)*inputs+feature;
    const float value=valid?input[i]:0;
    float rbf_dx=0;
    for (int g=0;g<grid;++g) {
        float dot=0;
        for (int start=0;start<outputs;start+=kTile) {
            const int ak=start+threadIdx.x, bk=start+threadIdx.y;
            dy[threadIdx.y][threadIdx.x]=kan_operand(row<rows && ak<outputs?upstream[static_cast<size_t>(row)*outputs+ak]:0,mode);
            w[threadIdx.y][threadIdx.x]=kan_operand(feature<inputs && bk<outputs?
                weight[(static_cast<size_t>(bk)*inputs+feature)*grid+g]:0,mode);
            __syncthreads();
            #pragma unroll
            for (int t=0;t<kTile;++t) dot=fmaf(dy[threadIdx.y][t],w[t][threadIdx.x],dot);
            __syncthreads();
        }
        if (valid) {
            const float width=fmaxf(widths[g],1e-3f);
            const float deriv=kan_basis(value,centers,widths,g)*(centers[g]-value)/(width*width);
            rbf_dx+=dot*deriv;
        }
    }
    if (valid) grad_input[i]+=rbf_dx;
}
bool kan_geometry(int rows,int inputs,int outputs,int grid,int mode) {
    return rows>0 && inputs>0 && outputs>0 && grid>=2 && mode>=0 && mode<=2 &&
        static_cast<long long>(inputs)*grid<=INT_MAX-15 &&
        static_cast<long long>(rows)*inputs<=INT_MAX &&
        static_cast<long long>(rows)*outputs<=INT_MAX &&
        static_cast<long long>(inputs)*grid*outputs<=INT_MAX &&
        rows<=65535*kTile && outputs<=65535*kTile;
}

#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
// Four native wave32 tiles share coalesced loads. Basis values exist only in
// LDS; dX contracts one grid point at a time and fuses its exact derivative.
template<int Kind,class DataT>
__global__ __launch_bounds__(128) void kan_rbf_wmma(
    const float* input,const float* operand,const float* weight,const float* centers,
    const float* widths,const float* base,const float* bias,float* output,
    int rows,int inputs,int outputs,int grid) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    constexpr int T=32;
    const int features=inputs*grid;
    const int m=Kind==1?outputs:rows, n=Kind==0?outputs:Kind==1?features:inputs;
    const int k=Kind==0?features:Kind==1?rows:outputs;
    const int rb=blockIdx.y*T, cb=blockIdx.x*T;
    __shared__ __align__(32) DataT left[T*T],right[T*T];
    __shared__ __align__(32) float result[T*T];
    const int wave=threadIdx.x/32, wr=(wave/2)*16, wc=(wave%2)*16;
    rocwmma::fragment<rocwmma::matrix_a,16,16,16,DataT,rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b,16,16,16,DataT,rocwmma::row_major> fb;
    rocwmma::fragment<rocwmma::accumulator,16,16,16,float> acc;
    float dx[8]={};
    for (int g=0;g<(Kind==2?grid:1);++g) {
        rocwmma::fill_fragment(acc,0.0f);
        for (int start=0;start<k;start+=T) {
            for (int i=threadIdx.x;i<T*T;i+=128) {
                const int r=i/T,c=i%T;
                float a=0,b=0;
                if constexpr (Kind==0) {
                    if (rb+r<rows && start+c<features)
                        a=kan_basis(input[static_cast<size_t>(rb+r)*inputs+(start+c)/grid],centers,widths,(start+c)%grid);
                    if (cb+r<outputs && start+c<features)
                        b=weight[static_cast<size_t>(cb+r)*features+start+c];
                } else if constexpr (Kind==1) {
                    if (rb+c<outputs && start+r<rows)
                        a=operand[static_cast<size_t>(start+r)*outputs+rb+c];
                    if (cb+c<features && start+r<rows)
                        b=kan_basis(input[static_cast<size_t>(start+r)*inputs+(cb+c)/grid],centers,widths,(cb+c)%grid);
                } else {
                    if (rb+r<rows && start+c<outputs)
                        a=operand[static_cast<size_t>(rb+r)*outputs+start+c];
                    if (cb+c<inputs && start+r<outputs)
                        b=weight[(static_cast<size_t>(start+r)*inputs+cb+c)*grid+g];
                }
                left[Kind==1?c*T+r:i]=static_cast<DataT>(a);
                right[Kind==0?c*T+r:i]=static_cast<DataT>(b);
            }
            __syncthreads();
            #pragma unroll
            for (int sub=0;sub<T;sub+=16) {
                rocwmma::load_matrix_sync(fa,left+wr*T+sub,T);
                rocwmma::load_matrix_sync(fb,right+sub*T+wc,T);
                rocwmma::mma_sync(acc,fa,fb,acc);
            }
            __syncthreads();
        }
        rocwmma::store_matrix_sync(result+wr*T+wc,acc,T,rocwmma::layout_t::mem_row_major);
        __syncthreads();
        for (int i=threadIdx.x;i<T*T;i+=128) {
            const int row=rb+i/T,col=cb+i%T;
            if (row>=m || col>=n) continue;
            const size_t index=static_cast<size_t>(row)*n+col;
            if constexpr (Kind==2) {
                const float value=input[index],width=fmaxf(widths[g],1e-3f);
                const float derivative=kan_basis(value,centers,widths,g)*(centers[g]-value)/(width*width);
                dx[i/128]+=result[i]*derivative;
            } else if constexpr (Kind==0) output[index]=(base[index]+result[i])+bias[col];
            else output[index]=result[i];
        }
        __syncthreads(); // all result readers finish before the next grid point
    }
    if constexpr (Kind==2) for (int i=threadIdx.x;i<T*T;i+=128) {
        const int row=rb+i/T,col=cb+i%T;
        if (row<m && col<n) output[static_cast<size_t>(row)*n+col]+=dx[i/128];
    }
#endif
}
template<int Kind,class DataT>
void kan_launch_wmma(const float* input,const float* operand,const float* weight,const float* centers,
    const float* widths,const float* base,const float* bias,float* output,int rows,int inputs,int outputs,int grid) {
    const int m=Kind==1?outputs:rows,n=Kind==0?outputs:Kind==1?inputs*grid:inputs;
    kan_rbf_wmma<Kind,DataT><<<dim3((n-1)/32+1,(m-1)/32+1),128,0,gpu::current_stream()>>>(
        input,operand,weight,centers,widths,base,bias,output,rows,inputs,outputs,grid);
}
#endif
template<int Kind>
bool kan_dispatch_wmma(const float* input,const float* operand,const float* weight,const float* centers,
    const float* widths,const float* base,const float* bias,float* output,int rows,int inputs,int outputs,int grid,int mode) {
    if (static_cast<long long>(inputs)*grid>INT_MAX-31 || !moe_training_wmma_supported()) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    if (mode==1) kan_launch_wmma<Kind,rocwmma::bfloat16_t>(input,operand,weight,centers,widths,base,bias,output,rows,inputs,outputs,grid);
    else kan_launch_wmma<Kind,rocwmma::float16_t>(input,operand,weight,centers,widths,base,bias,output,rows,inputs,outputs,grid);
    gpu::record_dispatch(gpu::DispatchPath::KanWmmaGemm);
    return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}

// One thread per (row, feature); each loops over the (small) grid.  This
// matches the CPU layout in src/kan.cpp::compute_basis exactly.
__global__ void kan_rbf_basis_forward_kernel(const float* __restrict__ x,
                                             const float* __restrict__ centers,
                                             const float* __restrict__ widths,
                                             float* __restrict__ basis, int rows,
                                             int input_dim, int grid) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * input_dim;
    if (idx >= total) return;

    const int feature = idx % input_dim;
    const int row = idx / input_dim;
    const float value = x[row * input_dim + feature];
    const int base = (row * input_dim + feature) * grid;

    for (int g = 0; g < grid; ++g) {
        const float w = fmaxf(widths[g], 1e-3f);  // safe_width
        const float diff = (value - centers[g]) / w;
        basis[base + g] = expf(-0.5f * diff * diff);
    }
}

// d(basis_g)/d(value) = basis_g * (center_g - value) / width_g^2.
// Accumulates the RBF contribution into grad_input (which already holds the
// base-weight gradient term), matching src/kan.cpp::backward.
__global__ void kan_rbf_basis_backward_kernel(const float* __restrict__ x,
                                              const float* __restrict__ grad_basis,
                                              const float* __restrict__ centers,
                                              const float* __restrict__ widths,
                                              float* __restrict__ grad_input,
                                              int rows, int input_dim, int grid) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * input_dim;
    if (idx >= total) return;

    const int feature = idx % input_dim;
    const int row = idx / input_dim;
    const float value = x[row * input_dim + feature];
    const int base = (row * input_dim + feature) * grid;

    float rbf_dx = 0.0f;
    for (int g = 0; g < grid; ++g) {
        const float w = fmaxf(widths[g], 1e-3f);
        const float diff = (value - centers[g]) / w;
        const float basis = expf(-0.5f * diff * diff);
        const float deriv = basis * (centers[g] - value) / (w * w);
        rbf_dx += grad_basis[base + g] * deriv;
    }
    grad_input[row * input_dim + feature] += rbf_dx;
}

}  // namespace

bool launch_kan_prepare_ternary(const float* weight,float* effective,float* scale,float* partials,int n) {
    if (!weight || !effective || !scale || !partials || n<=0) return false;
    const int blocks=n<4096*256?(n-1)/256+1:4096;
    const auto stream=gpu::current_stream();
    kan_qat_partials<<<blocks,256,0,stream>>>(weight,partials,n);
    kan_qat_scale<<<1,1,0,stream>>>(partials,scale,n,blocks);
    kan_qat_quantize<<<blocks,256,0,stream>>>(weight,effective,scale,n);
    return cudaGetLastError()==cudaSuccess;
}
bool launch_kan_rbf_projection(const float* input,const float* weight,const float* centers,
    const float* widths,const float* base,const float* bias,float* output,
    int rows,int inputs,int outputs,int grid,int mode,bool wmma) {
    if (!input || !weight || !centers || !widths || !base || !bias || !output ||
        !kan_geometry(rows,inputs,outputs,grid,mode)) return false;
    if (wmma && moe_training_wmma_geometry(rows,inputs,outputs,mode))
        return kan_dispatch_wmma<0>(input,nullptr,weight,centers,widths,base,bias,output,rows,inputs,outputs,grid,mode);
    kan_rbf_gemm<0><<<dim3((outputs+15)/16,(rows+15)/16),dim3(16,16),0,gpu::current_stream()>>>(
        input,weight,centers,widths,base,bias,output,rows,inputs,outputs,grid,mode);
    return cudaGetLastError()==cudaSuccess;
}
bool launch_kan_rbf_weight_backward(const float* input,const float* upstream,const float* centers,
    const float* widths,float* gradient,int rows,int inputs,int outputs,int grid,int mode,bool wmma) {
    if (!input || !upstream || !centers || !widths || !gradient ||
        !kan_geometry(rows,inputs,outputs,grid,mode)) return false;
    if (wmma && moe_training_wmma_geometry(rows,inputs,outputs,mode))
        return kan_dispatch_wmma<1>(input,upstream,nullptr,centers,widths,nullptr,nullptr,gradient,rows,inputs,outputs,grid,mode);
    kan_rbf_gemm<1><<<dim3((inputs*grid+15)/16,(outputs+15)/16),dim3(16,16),0,gpu::current_stream()>>>(
        input,upstream,centers,widths,nullptr,nullptr,gradient,rows,inputs,outputs,grid,mode);
    return cudaGetLastError()==cudaSuccess;
}
bool launch_kan_rbf_input_backward(const float* input,const float* upstream,const float* weight,
    const float* centers,const float* widths,float* gradient,
    int rows,int inputs,int outputs,int grid,int mode,bool wmma) {
    if (!input || !upstream || !weight || !centers || !widths || !gradient ||
        !kan_geometry(rows,inputs,outputs,grid,mode)) return false;
    if (wmma && moe_training_wmma_geometry(rows,inputs,outputs,mode))
        return kan_dispatch_wmma<2>(input,upstream,weight,centers,widths,nullptr,nullptr,gradient,rows,inputs,outputs,grid,mode);
    kan_rbf_dx<<<dim3((inputs+15)/16,(rows+15)/16),dim3(16,16),0,gpu::current_stream()>>>(
        input,upstream,weight,centers,widths,gradient,rows,inputs,outputs,grid,mode);
    return cudaGetLastError()==cudaSuccess;
}

void launch_kan_rbf_basis_forward(const float* x, const float* centers,
                                  const float* widths, float* basis, int rows,
                                  int input_dim, int grid) {
    if (rows <= 0 || input_dim <= 0) return;
    if (!x || !centers || !widths || !basis || grid < 2)
        throw std::invalid_argument("KAN basis requires valid device buffers and grid >= 2");
    if (rows > INT_MAX / input_dim || static_cast<long long>(rows) * input_dim > INT_MAX / grid) {
        throw std::overflow_error("KAN basis grid exceeds INT_MAX elements");
    }
    const int total = rows * input_dim;
    const int blocks = gpu::ceil_div_positive(total, kKanThreads);
    kan_rbf_basis_forward_kernel<<<blocks, kKanThreads, 0, nsos::gpu::current_stream()>>>(
        x, centers, widths, basis, rows, input_dim, grid);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("KAN materialized basis forward launch failed");
}

void launch_kan_rbf_basis_backward(const float* x, const float* grad_basis,
                                   const float* centers, const float* widths,
                                   float* grad_input, int rows, int input_dim,
                                   int grid) {
    if (rows <= 0 || input_dim <= 0) return;
    if (!x || !grad_basis || !centers || !widths || !grad_input || grid < 2)
        throw std::invalid_argument("KAN basis backward requires valid device buffers and grid >= 2");
    if (rows > INT_MAX / input_dim || static_cast<long long>(rows) * input_dim > INT_MAX / grid) {
        throw std::overflow_error("KAN basis grid exceeds INT_MAX elements");
    }
    const int total = rows * input_dim;
    const int blocks = gpu::ceil_div_positive(total, kKanThreads);
    kan_rbf_basis_backward_kernel<<<blocks, kKanThreads, 0, nsos::gpu::current_stream()>>>(
        x, grad_basis, centers, widths, grad_input, rows, input_dim, grid);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("KAN materialized basis backward launch failed");
}

}  // namespace cuda
}  // namespace nsos

#endif  // USE_CUDA
