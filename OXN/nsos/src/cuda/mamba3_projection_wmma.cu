#define NSOS_INCLUDE_ROCWMMA 1
#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/mamba3_projection_wmma.cuh"

namespace nsos::mamba3_projection {
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
namespace {
constexpr int Tile=32, Threads=128;
template<bool TA,bool TB,class Low>
__global__ __launch_bounds__(Threads) void dense_wmma(
    const float* a,const float* b,float* output,int rows,int cols,int reduction,int axis,int begin) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    const int row_base=blockIdx.y*Tile,col_base=blockIdx.x*Tile;
    __shared__ __align__(32) Low left[Tile*Tile],right[Tile*Tile];
    __shared__ __align__(32) float result[Tile*Tile];
    const int wave=threadIdx.x/32,wr=(wave/2)*16,wc=(wave%2)*16;
    rocwmma::fragment<rocwmma::matrix_a,16,16,16,Low,rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b,16,16,16,Low,rocwmma::row_major> fb;
    rocwmma::fragment<rocwmma::accumulator,16,16,16,float> acc;
    rocwmma::fill_fragment(acc,0.0f);
    for(int start=0;start<reduction;start+=Tile) {
        for(int i=threadIdx.x;i<Tile*Tile;i+=Threads) {
            const int r=i/Tile,c=i%Tile;
            // Transpose coalesced global loads in LDS; all padded operands are
            // explicitly zero, including K tails. Every thread hits barriers.
            float av=0,bv=0;
            if constexpr(TA) {
                if(row_base+c<rows && start+r<reduction)
                    av=a[std::size_t(start+r)*rows+row_base+c];
            } else {
                if(row_base+r<rows && start+c<reduction)
                    av=a[std::size_t(row_base+r)*reduction+start+c];
            }
            if constexpr(TB) {
                if(col_base+r<cols && start+c<reduction)
                    bv=b[std::size_t(col_base+r)*reduction+start+c];
            } else {
                if(col_base+c<cols && start+r<reduction)
                    bv=b[std::size_t(start+r)*cols+col_base+c];
            }
            const int ar=row_base+(TA?c:r),ak=start+(TA?r:c);
            const int bk=start+(TB?c:r),bc=col_base+(TB?r:c);
            if((axis==1 && ar>=begin)||(axis==3 && ak>=begin)) av=0;
            if((axis==2 && bc>=begin)||(axis==3 && bk>=begin)) bv=0;
            left[TA?c*Tile+r:i]=static_cast<Low>(av);
            right[TB?c*Tile+r:i]=static_cast<Low>(bv);
        }
        __syncthreads();
        #pragma unroll
        for(int sub=0;sub<Tile;sub+=16) {
            rocwmma::load_matrix_sync(fa,left+wr*Tile+sub,Tile);
            rocwmma::load_matrix_sync(fb,right+sub*Tile+wc,Tile);
            rocwmma::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads();
    }
    rocwmma::store_matrix_sync(result+wr*Tile+wc,acc,Tile,rocwmma::layout_t::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<Tile*Tile;i+=Threads) {
        const int row=row_base+i/Tile,col=col_base+i%Tile;
        if(row<rows && col<cols) output[std::size_t(row)*cols+col]=result[i];
    }
#endif
}
// Eight waves reuse an FP32 operand tile across 32x8 output elements.
// Padding removes power-of-two LDS row strides. Each lane still accumulates
// exactly K=begin+lane, begin+lane+32, ... and uses the original shuffle tree:
// controltail_fp32_v1 arithmetic and checkpoint identity remain unchanged.
constexpr int TailRows=32, TailCols=8, TailK=32, TailThreads=256;
template<bool TA,bool TB>
__global__ __launch_bounds__(TailThreads) void exact_tail(const float* a,const float* b,float* output,
    int rows,int cols,int k,int axis,int begin) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    const int row_base=(axis==1?begin:0)+int(blockIdx.y)*TailRows;
    const int col_base=(axis==2?begin:0)+int(blockIdx.x)*TailCols;
    const int lane=threadIdx.x%32,wave=threadIdx.x/32;
    __shared__ float left[TailRows][TailK+1],right[TailK][TailCols+1];
    float sums[4][TailCols]={};
    for(int start=axis==3?begin:0;start<k;start+=TailK) {
        for(int i=threadIdx.x;i<TailRows*TailK;i+=TailThreads) {
            // Select the physical contiguous axis before transposing in LDS.
            const int r=TA?i%TailRows:i/TailK;
            const int x=TA?i/TailRows:i%TailK;
            const int row=row_base+r,kk=start+x;
            left[r][x]=row<rows && kk<k?
                a[TA?std::size_t(kk)*rows+row:std::size_t(row)*k+kk]:0.f;
        }
        for(int i=threadIdx.x;i<TailK*TailCols;i+=TailThreads) {
            const int x=TB?i%TailK:i/TailCols;
            const int c=TB?i/TailK:i%TailCols;
            const int kk=start+x,col=col_base+c;
            right[x][c]=kk<k && col<cols?
                b[TB?std::size_t(col)*k+kk:std::size_t(kk)*cols+col]:0.f;
        }
        __syncthreads();
        // Do not introduce a zero FMA on padded K lanes: their old sums stay
        // untouched, which also preserves signed-zero and cancellation behavior.
        if(start+lane<k) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                const float av=left[wave+8*r][lane];
                #pragma unroll
                for(int c=0;c<TailCols;++c)
                    sums[r][c]=fmaf(av,right[lane][c],sums[r][c]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int c=0;c<TailCols;++c) {
            float sum=sums[r][c];
            for(int delta=16;delta;delta/=2) sum+=__shfl_down(sum,delta,32);
            const int row=row_base+wave+8*r,col=col_base+c;
            if(lane==0 && row<rows && col<cols) {
                const auto position=std::size_t(row)*cols+col;
                output[position]=axis==3?output[position]+sum:sum;
            }
        }
    }
#endif
}
template<bool TA,bool TB,class Low> void enqueue(dim3 grid,int rows,int cols,int k,
    const float* a,const float* b,float* output,int axis,int begin) {
    const auto stream=gpu::current_stream();
    dense_wmma<TA,TB,Low><<<grid,Threads,0,stream>>>(a,b,output,rows,cols,k,axis,begin);
    if(axis) {
        const int tail_rows=axis==1?rows-begin:rows;
        const int tail_cols=axis==2?cols-begin:cols;
        const dim3 tail_grid((tail_cols-1)/TailCols+1,(tail_rows-1)/TailRows+1);
        exact_tail<TA,TB><<<tail_grid,TailThreads,0,stream>>>(a,b,output,rows,cols,k,axis,begin);
    }
}
template<class Low> bool kernel_supported(const hipDeviceProp_t& prop) {
    for(auto fn:{dense_wmma<false,false,Low>,dense_wmma<false,true,Low>,
        dense_wmma<true,false,Low>,dense_wmma<true,true,Low>}) {
        hipFuncAttributes attrs{};
        if(hipFuncGetAttributes(&attrs,reinterpret_cast<const void*>(fn))!=hipSuccess || attrs.maxThreadsPerBlock<Threads ||
            attrs.sharedSizeBytes>prop.sharedMemPerBlock) return false;
    }
    for(auto fn:{exact_tail<false,false>,exact_tail<false,true>,exact_tail<true,false>,exact_tail<true,true>}) {
        hipFuncAttributes attrs{};
        if(hipFuncGetAttributes(&attrs,reinterpret_cast<const void*>(fn))!=hipSuccess || attrs.maxThreadsPerBlock<TailThreads ||
            attrs.sharedSizeBytes>prop.sharedMemPerBlock) return false;
    }
    return true;
}
template<class Low> void launch(bool ta,bool tb,int rows,int cols,int k,
    const float* a,const float* b,float* output,int axis,int begin) {
    const dim3 grid((cols-1)/Tile+1,(rows-1)/Tile+1);
    if(ta) {
        if(tb) enqueue<true,true,Low>(grid,rows,cols,k,a,b,output,axis,begin);
        else enqueue<true,false,Low>(grid,rows,cols,k,a,b,output,axis,begin);
    } else {
        if(tb) enqueue<false,true,Low>(grid,rows,cols,k,a,b,output,axis,begin);
        else enqueue<false,false,Low>(grid,rows,cols,k,a,b,output,axis,begin);
    }
}
}
#endif
bool supported(Policy mode) {
    if(mode!=Policy::BF16 && mode!=Policy::FP16) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    int device=-1;if(hipGetDevice(&device)!=hipSuccess) return false;
    thread_local int cached_device=-1;
    thread_local bool cached_bf16=false,cached_fp16=false;
    if(cached_device==device) return mode==Policy::BF16?cached_bf16:cached_fp16;
    hipDeviceProp_t prop{};if(hipGetDeviceProperties(&prop,device)!=hipSuccess) return false;
    const auto arch=std::string(prop.gcnArchName).substr(0,std::string(prop.gcnArchName).find(':'));
    if(prop.warpSize!=32 || (arch!="gfx1100" && arch!="gfx1101" && arch!="gfx1102")) return false;
    bool compiled=false;
    for(const auto& info:gpu::enumerate_devices()) if(info.index==device) compiled=info.compiled;
    if(!compiled) return false;
    cached_bf16=kernel_supported<rocwmma::bfloat16_t>(prop);
    cached_fp16=kernel_supported<rocwmma::float16_t>(prop);cached_device=device;
    return mode==Policy::BF16?cached_bf16:cached_fp16;
#else
    return false;
#endif
}
bool gemm(Policy mode,bool ta,bool tb,int rows,int cols,int k,
    const float* a,const float* b,float* output,ExactAxis exact_axis,int exact_begin) {
    const int axis=int(exact_axis),begin=exact_begin;
    const int extent=axis==1?rows:axis==2?cols:k;
    if(axis<0 || axis>3 || (axis && (begin<0 || begin>=extent))) return false;
    if(!a || !b || !output || !geometry(rows,cols,k) || !supported(mode)) return false;
    // Reject partial in-place ranges as well as identical base pointers.
    const auto ao=reinterpret_cast<std::uintptr_t>(a),bo=reinterpret_cast<std::uintptr_t>(b),co=reinterpret_cast<std::uintptr_t>(output);
    const std::size_t as=std::size_t(rows)*k*sizeof(float),bs=std::size_t(cols)*k*sizeof(float),cs=std::size_t(rows)*cols*sizeof(float);
    const auto maximum=~std::uintptr_t(0);
    if(ao>maximum-as || bo>maximum-bs || co>maximum-cs) return false;
    if((co<ao+as && ao<co+cs)||(co<bo+bs && bo<co+cs)) return false;
#if defined(NSOS_GPU_BACKEND_HIP) && defined(NSOS_HAS_ROCWMMA)
    if(mode==Policy::BF16) launch<rocwmma::bfloat16_t>(ta,tb,rows,cols,k,a,b,output,axis,begin);
    else launch<rocwmma::float16_t>(ta,tb,rows,cols,k,a,b,output,axis,begin);
    if(hipGetLastError()!=hipSuccess) return false;
    gpu::record_dispatch(gpu::DispatchPath::Mamba3ProjectionWmma);
    return true;
#else
    return false;
#endif
}
}
