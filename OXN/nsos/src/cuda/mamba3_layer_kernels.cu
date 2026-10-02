#include "gpu_backend.h"
#include "mamba3_layer_math.h"
#include "cuda/kernels.cuh"
#include "cuda/gpu_utils.h"
#include "gpu_execution.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

#include "mamba3_parallel_kernels.cuh"
#include "mamba3_hierarchical_kernels.cuh"

namespace nsos::mamba3_block {
namespace {
constexpr int threads=128;
__global__ void forward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,State<const float> initial,State<float> final,Trace<float> trace,float* y,float* status) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<s.batch&&!status[b]) detail::forward_batch(s,l,b,p,w,valid,initial,final,trace,y,status);
}
__global__ void backward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,State<const float> initial,Trace<const float> trace,const float* dy,State<const float> seed,float* dx,float* partial,State<float> di,float* scratch,float* status) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<s.batch) {
        const auto stride=std::size_t(s.head_dim)*s.state_dim+3*s.rank*s.state_dim+2*s.rank*s.head_dim+s.rotary_pairs;
        detail::backward_batch(s,l,b,p,w,valid,initial,trace,dy,seed,dx,partial+std::size_t(b)*l.total,di,scratch+std::size_t(b)*stride,status);
    }
}
__global__ void reduce_kernel(int batch,std::size_t n,const float* p,float* out,const float* status) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;float v=0;bool bad=false;
    for(int b=0;b<batch;++b) {bad|=status[b]!=0;v+=p[std::size_t(b)*n+i];}out[i]=bad?0:v;
}
__global__ void mask_kernel(int B,int S,int W,const int* valid,const float* src,float* dst) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=std::size_t(B)*S*W;
    if(i<total) {const int b=int(i/(std::size_t(S)*W)),t=int((i/W)%S);dst[i]=t<valid[b]?src[i]:0;}
}
__global__ void check_kernel(int B,int S,int W,const int* valid,const float* src,float* status) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=std::size_t(B)*S*W;
    // Validate every prefix once, including entirely masked batches. Invalid
    // prefixes never authorize a source read or get replaced by finite errors.
    if(i<std::size_t(B)) {const int len=valid[i];if(len<0||len>S) atomicExch(status+i,1.f);}
    if(i>=total) return;
    const int b=int(i/(std::size_t(S)*W)),t=int((i/W)%S),len=valid[b];
    if(len<0||len>S||t>=len) return; // poison padding is not read
    if(!isfinite(src[i])) atomicExch(status+b,3.f);
}
__global__ void gate_kernel(int B,std::size_t n,const float* status,float* dst,bool whole) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=std::size_t(B)*n) return;
    bool bad=status[i/n]!=0;if(whole) for(int b=0;b<B;++b) bad|=status[b]!=0;if(bad) dst[i]=0;
}
__global__ void parameter_check_kernel(int B,std::size_t n,const float* src,float* status) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    __shared__ int bad;
    if(!threadIdx.x) bad=0;
    __syncthreads();
    if(i<n&&!isfinite(src[i])) atomicExch(&bad,1);
    __syncthreads();
    // Coalesce failures within a block; shared parameters poison every batch.
    // The status is sticky: a successful block never clears another failure.
    if(!threadIdx.x&&bad) for(int b=0;b<B;++b) atomicExch(status+b,3.f);
}
__global__ void parameter_gate_kernel(int B,std::size_t n,const float* status,float* dst) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;bool bad=false;for(int b=0;b<B;++b) bad|=status[b]!=0;if(bad) dst[i]=0;
}
int blocks(std::size_t n) {
    const auto count=n/threads+(n%threads!=0);
    if(count>std::size_t(INT_MAX)) throw std::overflow_error("Mamba3 launch grid exceeds INT_MAX");
    return static_cast<int>(count);
}
// Host work is O(log_32 Q); no device lane walks Q chunks sequentially.
bool launch_hierarchy(Shape s,Layout l,const float* p,const float* w,const int* valid,
    State<const float> initial,Trace<const float> tr,State<const float> seed,
    BackwardWorkspace ws,bool reverse,float* boundaries,float* final_ssm,float* status) {
    namespace hd=hierarchical_detail;
    const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const auto stream=gpu::current_stream();
    int lengths[8]{},levels=1;std::size_t offsets[8]{};lengths[0]=(s.sequence+31)/32;
    while(lengths[levels-1]>32) {
        if(levels==8)return false;
        offsets[levels]=offsets[levels-1]+cells*lengths[levels-1];
        lengths[levels]=(lengths[levels-1]+31)/32;++levels;
    }
    const auto summary_warps=cells*lengths[0];
    hd::summary_kernel<<<blocks(summary_warps*32),128,0,stream>>>(s,l,p,w,valid,initial,tr,ws,reverse,tr.hierarchy_a,tr.hierarchy_b,status);
    for(int level=0;level<levels;++level) {
        const auto groups=(lengths[level]+31)/32;
        hd::level_kernel<<<blocks(cells*groups*32),128,0,stream>>>(s,lengths[level],offsets[level],
            level+1<levels?offsets[level+1]:0,level+1<levels,tr.hierarchy_a,tr.hierarchy_b,status);
    }
    for(int level=levels-2;level>=0;--level)
        hd::fixup_kernel<<<blocks(cells*lengths[level]),128,0,stream>>>(s,lengths[level],offsets[level],offsets[level+1],tr.hierarchy_a,tr.hierarchy_b,status);
    hd::publish_kernel<<<blocks(cells*(lengths[0]+1)),128,0,stream>>>(s,valid,seed,reverse,tr.hierarchy_a,tr.hierarchy_b,boundaries,final_ssm,status);
    return cudaGetLastError()==cudaSuccess;
}
bool state_complete(State<const float> s) {return s.phase&&s.ssm&&s.k&&s.v;}
bool state_complete(State<float> s) {return s.phase&&s.ssm&&s.k&&s.v;}
bool state_empty(State<const float> s) {return !s.phase&&!s.ssm&&!s.k&&!s.v;}
}
bool gpu_forward(Shape s,const float* p,const float* w,const int* valid,State<const float> initial,State<float> final,Trace<float> tr,float* y,float* status) {
    if(!eligible(s)||!p||!w||!valid||!state_complete(final)||(!state_empty(initial)&&!state_complete(initial))||!tr.history||!tr.q||!tr.k||!tr.phase||!tr.readout||!y||!status) return false;
    if(tr.checkpoints&&!tr.parallel) return false;
    if(tr.replay_lds&&(!tr.parallel||!tr.checkpoints)) return false;
    if(tr.hierarchical&&(!tr.parallel||!tr.checkpoints||!tr.replay_lds||!hierarchical_eligible(s)||!tr.hierarchy_a||!tr.hierarchy_b)) return false;
    if(tr.parallel) {
        if(!parallel_eligible(s)||!tr.coefficients) return false;
        int device=-1;cudaDeviceProp properties{};
        if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess||properties.sharedMemPerBlock<(tr.replay_lds?sizeof(parallel_detail::FlashBackwardSharedMaximum):16384)||properties.maxThreadsPerBlock<128) return false;
        const Layout l(s);const auto stream=gpu::current_stream();
        parameter_check_kernel<<<blocks(l.total),threads,0,stream>>>(s.batch,l.total,w,status);
        check_kernel<<<blocks(std::size_t(s.batch)*s.sequence*s.width()),threads,0,stream>>>(s.batch,s.sequence,s.width(),valid,p,status);
        parallel_detail::coefficient_kernel<<<blocks(std::size_t(s.batch)*s.heads*s.sequence),threads,0,stream>>>(s,l,p,w,valid,tr.coefficients,status);
        parallel_detail::empty_state_kernel<<<s.batch*s.heads,128,0,stream>>>(s,valid,initial,final,status);
        parallel_detail::phase_kernel<<<s.batch*s.heads,128,0,stream>>>(s,l,p,w,valid,initial,final,tr,status);
        parallel_detail::rotate_kernel<<<s.batch*s.heads*s.sequence,128,0,stream>>>(s,l,p,w,valid,initial,final,tr,status);
        const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
        if(tr.hierarchical) {
            if(!launch_hierarchy(s,l,p,w,valid,initial,parallel_detail::constant(tr),initial,{},false,tr.history,final.ssm,status))return false;
        } else parallel_detail::state_kernel<<<blocks(cells*32),128,0,stream>>>(s,l,p,w,valid,initial,final,tr,status);
        if(tr.checkpoints) {
            const auto readout_grid=std::size_t(s.batch)*s.heads*((s.sequence+parallel_detail::flash_readout_t-1)/parallel_detail::flash_readout_t)*((s.head_dim+parallel_detail::flash_readout_p-1)/parallel_detail::flash_readout_p);
            if(readout_grid>std::size_t(INT_MAX)) return false;
            parallel_detail::flash_readout_kernel<<<static_cast<int>(readout_grid),threads,0,stream>>>(s,l,p,w,valid,initial,tr,status);
        } else parallel_detail::readout_kernel<<<blocks(readout_size(s)),128,0,stream>>>(s,l,p,w,valid,initial,tr,status);
        parallel_detail::output_kernel<<<s.batch*s.heads*s.sequence,128,0,stream>>>(s,l,p,w,valid,parallel_detail::constant(tr),y,status);
        return cudaGetLastError()==cudaSuccess;
    }
    // Low occupancy serial baseline: intentional correctness implementation.
    // No inherited RDNA-only restriction; ordinary FP32/FP64 device arithmetic.
    forward_kernel<<<blocks(s.batch),threads,0,gpu::current_stream()>>>(s,Layout(s),p,w,valid,initial,final,tr,y,status);
    return cudaGetLastError()==cudaSuccess;
}
bool gpu_backward(Shape s,const float* p,const float* w,const int* valid,State<const float> initial,Trace<const float> tr,const float* dy,State<const float> seed,float* dx,float* partial,State<float> di,float* scratch,float* status,BackwardWorkspace ws) {
    if(!eligible(s)||!p||!w||!valid||!dy||!dx||!partial||!scratch||!status||!state_complete(di)||(!state_empty(initial)&&!state_complete(initial))||(!state_empty(seed)&&!state_complete(seed))||!tr.history||!tr.q||!tr.k||!tr.phase||!tr.readout) return false;
    if(tr.checkpoints&&!tr.parallel) return false;
    if(tr.replay_lds&&(!tr.parallel||!tr.checkpoints)) return false;
    if(tr.hierarchical&&(!tr.parallel||!tr.checkpoints||!tr.replay_lds||!hierarchical_eligible(s)||!tr.hierarchy_a||!tr.hierarchy_b)) return false;
    if(tr.parallel) {
        if(!parallel_eligible(s)||!tr.coefficients||!ws.gy||!ws.token_parameters||!ws.bc||!ws.phase||!ws.reverse||!ws.dz||!ws.dx||!ws.ddt||!ws.da||!ws.dtrap) return false;
        if(tr.replay_lds) {
            int device=-1;cudaDeviceProp properties{};
            if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess||properties.sharedMemPerBlock<sizeof(parallel_detail::FlashBackwardSharedMaximum)||properties.maxThreadsPerBlock<128) return false;
        }
        const Layout l(s);const auto stream=gpu::current_stream();
        parallel_detail::output_backward_kernel<<<s.batch*s.heads*s.sequence,128,0,stream>>>(s,l,p,w,valid,tr,dy,ws,status);
        const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
        if(tr.hierarchical) {
            if(!launch_hierarchy(s,l,p,w,valid,initial,tr,seed,ws,true,ws.reverse,di.ssm,status))return false;
        } else parallel_detail::reverse_kernel<<<blocks(cells*32),128,0,stream>>>(s,l,p,w,valid,tr,seed,di,ws,status);
        if(tr.replay_lds) {
            const auto grid=std::size_t(s.batch)*s.heads*((s.sequence+parallel_detail::flash_backward_t-1)/parallel_detail::flash_backward_t);
            if(grid>std::size_t(INT_MAX)) return false;
            if(s.rank==1) parallel_detail::flash_token_backward_kernel<1><<<static_cast<int>(grid),128,0,stream>>>(s,l,p,w,valid,initial,tr,seed,di,ws,status);
            else if(s.rank==2) parallel_detail::flash_token_backward_kernel<2><<<static_cast<int>(grid),128,0,stream>>>(s,l,p,w,valid,initial,tr,seed,di,ws,status);
            else if(s.rank<=4) parallel_detail::flash_token_backward_kernel<4><<<static_cast<int>(grid),128,0,stream>>>(s,l,p,w,valid,initial,tr,seed,di,ws,status);
            else parallel_detail::flash_token_backward_kernel<8><<<static_cast<int>(grid),128,0,stream>>>(s,l,p,w,valid,initial,tr,seed,di,ws,status);
        } else
        parallel_detail::token_backward_kernel<<<s.batch*s.heads*s.sequence,128,0,stream>>>(s,l,p,w,valid,initial,tr,seed,di,ws,status);
        parallel_detail::phase_backward_kernel<<<s.batch*s.heads*s.rotary_pairs,32,0,stream>>>(s,l,p,w,valid,seed,di,ws,status);
        parallel_detail::gather_projection_kernel<<<blocks(std::size_t(s.batch)*s.sequence*s.width()),128,0,stream>>>(s,l,p,w,valid,tr,ws,dx,status);
        parallel_detail::gather_parameters_kernel<<<blocks(std::size_t(s.batch)*l.total),128,0,stream>>>(s,l,valid,ws,dx,partial,status);
        parallel_detail::empty_backward_kernel<<<s.batch*s.heads,128,0,stream>>>(s,valid,seed,di,status);
        return cudaGetLastError()==cudaSuccess;
    }
    backward_kernel<<<blocks(s.batch),threads,0,gpu::current_stream()>>>(s,Layout(s),p,w,valid,initial,tr,dy,seed,dx,partial,di,scratch,status);
    return cudaGetLastError()==cudaSuccess;
}
bool gpu_reduce(Shape s,const float* p,float* out,const float* status) {
    if(!eligible(s)||!p||!out||!status) return false;
    reduce_kernel<<<blocks(Layout(s).total),threads,0,gpu::current_stream()>>>(s.batch,Layout(s).total,p,out,status);return cudaGetLastError()==cudaSuccess;
}
bool gpu_mask(int B,int S,int W,const int* valid,const float* src,float* dst) {
    if(B<=0||S<=0||W<=0||!valid||!src||!dst||std::size_t(B)*S*W>INT_MAX) return false;
    mask_kernel<<<blocks(std::size_t(B)*S*W),threads,0,gpu::current_stream()>>>(B,S,W,valid,src,dst);return cudaGetLastError()==cudaSuccess;
}
bool gpu_check(int B,int S,int W,const int* valid,const float* src,float* status) {
    if(B<=0||S<=0||W<=0||!valid||!src||!status) return false;
    // Tensor element capacity and division guards prevent overflow in both
    // the launch geometry and the kernel's row/prefix address calculation.
    const auto cap=std::size_t(INT_MAX);
    if(std::size_t(B)>cap/std::size_t(S)||std::size_t(B)*S>cap/std::size_t(W)) return false;
    const auto total=std::size_t(B)*S*W;
    check_kernel<<<blocks(total),threads,0,gpu::current_stream()>>>(B,S,W,valid,src,status);return cudaGetLastError()==cudaSuccess;
}
bool gpu_gate(int B,std::size_t n,const float* status,float* dst,bool whole) {
    if(B<=0||!n||n>std::size_t(INT_MAX)/B||!status||!dst) return false;
    gate_kernel<<<blocks(std::size_t(B)*n),threads,0,gpu::current_stream()>>>(B,n,status,dst,whole);return cudaGetLastError()==cudaSuccess;
}
bool gpu_parameter_check(int B,std::size_t n,const float* src,float* status) {
    if(B<=0||B>65535||!n||n>INT_MAX||!src||!status) return false;
    parameter_check_kernel<<<blocks(n),threads,0,gpu::current_stream()>>>(B,n,src,status);return cudaGetLastError()==cudaSuccess;
}
bool gpu_parameter_gate(int B,std::size_t n,const float* status,float* dst) {
    if(B<=0||B>65535||!n||n>INT_MAX||!status||!dst) return false;
    parameter_gate_kernel<<<blocks(n),threads,0,gpu::current_stream()>>>(B,n,status,dst);return cudaGetLastError()==cudaSuccess;
}
}
