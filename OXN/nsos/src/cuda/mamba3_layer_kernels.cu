#include "gpu_backend.h"
#include "mamba3_layer_math.h"
#include "cuda/kernels.cuh"
#include "cuda/gpu_utils.h"
#include "gpu_execution.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

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
    const int b=blockIdx.x;if(threadIdx.x||b>=B) return;
    if(valid[b]<0||valid[b]>S) {status[b]=1;return;}
    for(std::size_t i=0;i<std::size_t(valid[b])*W;++i) if(!isfinite(src[std::size_t(b)*S*W+i])) {status[b]=3;return;}
}
__global__ void gate_kernel(int B,std::size_t n,const float* status,float* dst,bool whole) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=std::size_t(B)*n) return;
    bool bad=status[i/n]!=0;if(whole) for(int b=0;b<B;++b) bad|=status[b]!=0;if(bad) dst[i]=0;
}
__global__ void parameter_check_kernel(int B,std::size_t n,const float* src,float* status) {
    if(blockIdx.x||threadIdx.x) return;bool bad=false;for(std::size_t i=0;i<n;++i) bad|=!isfinite(src[i]);if(bad) for(int b=0;b<B;++b) status[b]=3;
}
__global__ void parameter_gate_kernel(int B,std::size_t n,const float* status,float* dst) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n) return;bool bad=false;for(int b=0;b<B;++b) bad|=status[b]!=0;if(bad) dst[i]=0;
}
int blocks(std::size_t n) {
    const auto count=n/threads+(n%threads!=0);
    if(count>std::size_t(INT_MAX)) throw std::overflow_error("Mamba3 launch grid exceeds INT_MAX");
    return static_cast<int>(count);
}
bool state_complete(State<const float> s) {return s.phase&&s.ssm&&s.k&&s.v;}
bool state_complete(State<float> s) {return s.phase&&s.ssm&&s.k&&s.v;}
bool state_empty(State<const float> s) {return !s.phase&&!s.ssm&&!s.k&&!s.v;}
}
bool gpu_forward(Shape s,const float* p,const float* w,const int* valid,State<const float> initial,State<float> final,Trace<float> tr,float* y,float* status) {
    if(!eligible(s)||!p||!w||!valid||!state_complete(final)||(!state_empty(initial)&&!state_complete(initial))||!tr.history||!tr.q||!tr.k||!tr.phase||!tr.readout||!y||!status) return false;
    // Low occupancy serial baseline: intentional correctness implementation.
    // No inherited RDNA-only restriction; ordinary FP32/FP64 device arithmetic.
    forward_kernel<<<blocks(s.batch),threads,0,gpu::current_stream()>>>(s,Layout(s),p,w,valid,initial,final,tr,y,status);
    return cudaGetLastError()==cudaSuccess;
}
bool gpu_backward(Shape s,const float* p,const float* w,const int* valid,State<const float> initial,Trace<const float> tr,const float* dy,State<const float> seed,float* dx,float* partial,State<float> di,float* scratch,float* status) {
    if(!eligible(s)||!p||!w||!valid||!dy||!dx||!partial||!scratch||!status||!state_complete(di)||(!state_empty(initial)&&!state_complete(initial))||(!state_empty(seed)&&!state_complete(seed))||!tr.history||!tr.q||!tr.k||!tr.phase||!tr.readout) return false;
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
    check_kernel<<<B,1,0,gpu::current_stream()>>>(B,S,W,valid,src,status);return cudaGetLastError()==cudaSuccess;
}
bool gpu_gate(int B,std::size_t n,const float* status,float* dst,bool whole) {
    if(B<=0||!n||n>std::size_t(INT_MAX)/B||!status||!dst) return false;
    gate_kernel<<<blocks(std::size_t(B)*n),threads,0,gpu::current_stream()>>>(B,n,status,dst,whole);return cudaGetLastError()==cudaSuccess;
}
bool gpu_parameter_check(int B,std::size_t n,const float* src,float* status) {
    if(B<=0||B>65535||!n||n>INT_MAX||!src||!status) return false;
    parameter_check_kernel<<<1,1,0,gpu::current_stream()>>>(B,n,src,status);return cudaGetLastError()==cudaSuccess;
}
bool gpu_parameter_gate(int B,std::size_t n,const float* status,float* dst) {
    if(B<=0||B>65535||!n||n>INT_MAX||!status||!dst) return false;
    parameter_gate_kernel<<<blocks(n),threads,0,gpu::current_stream()>>>(B,n,status,dst);return cudaGetLastError()==cudaSuccess;
}
}
