#include "gpu_backend.h"
#include "cuda/tiled_cross_entropy.cuh"
#include <cmath>
#include <stdexcept>
namespace {
__global__ void mask_kernel(float* x,const float* m,int rows,int cols) {
    size_t ix=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(ix>=static_cast<size_t>(rows)*cols)return;
    const float* mr=m+(ix/cols)*7;if(mr[1]==0 && mr[2]==0 && mr[3]<0)x[ix]=0;
}
__global__ void statistics_kernel(const float* z,float* s,const float* m,int rows,int cols,int start) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    float* sr=s+r*12; const float* mr=m+r*7;
    if(mr[1]==0 && mr[2]==0 && mr[3]<0) return;
    float mx=sr[0]; for(int c=0;c<cols;++c) mx=fmaxf(mx,z[r*cols+c]);
    double sum=sr[1]==0 ? 0 : sr[1]*exp(static_cast<double>(sr[0]-mx)); double sq=sr[7];
    for(int c=0;c<cols;++c) {
        float v=z[r*cols+c]; int id=start+c; sum+=exp(static_cast<double>(v-mx)); sq+=static_cast<double>(v)*v;
        if(id==static_cast<int>(mr[0])) sr[2]=v;
        for(int k=0;k<4;++k) if(id==static_cast<int>(mr[3+k])) sr[3+k]=v;
    }
    sr[0]=mx; sr[1]=static_cast<float>(sum); sr[7]=static_cast<float>(sq);
}
__global__ void finish_kernel(float* s,float* l,const float* m,int rows,int vocab,float rul,float beta) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
    float* sr=s+r*12; const float* mr=m+r*7; if(sr[1]==0) return;
    if(mr[1]>0)l[r*3]=mr[1]*((sr[0]-sr[2])+logf(sr[1]));
    if(beta>0 && mr[2]>0)l[r*3+2]=0.5f*beta*mr[2]*sr[7]/vocab;
    for(int k=0;k<4;++k) {
        float p=mr[3+k]<0 ? 0 : expf(sr[3+k]-sr[0])/sr[1]; sr[3+k]=0;
        if(p>1e-6f && p<1.0f-1e-6f) {sr[3+k]=rul*p/fmaxf(1.0f-p,1e-6f);l[r*3+1]-=rul*log1pf(-p);}
    }
}
__global__ void gradient_kernel(const float* z,float* out,const float* s,const float* m,
 int rows,int cols,int start,int vocab,float beta,float scale) {
    size_t ix=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(ix>=static_cast<size_t>(rows)*cols) return; int r=static_cast<int>(ix/cols), id=start+static_cast<int>(ix%cols);
    const float* sr=s+r*12; const float* mr=m+r*7; if(sr[1]==0) return;
    float factor=0;for(int k=0;k<4;++k) factor+=sr[3+k];
    float p=expf(z[ix]-sr[0])/sr[1]; float value=mr[1]*(p-(id==static_cast<int>(mr[0]) ? 1.0f : 0.0f))-factor*p;
    for(int k=0;k<4;++k) if(id==static_cast<int>(mr[3+k])) value+=sr[3+k];
    out[ix]=scale*(value+beta*mr[2]*z[ix]/vocab);
}
__global__ void add_kernel(float* dst,const float* src,int rows,int cols,int stride,int row,int col) {
    size_t ix=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(ix>=static_cast<size_t>(rows)*cols) return;
    dst[(ix/cols+row)*stride+ix%cols+col]+=src[ix];
}
void check() {auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
}
extern "C" void launch_cce_mask(float* x,const float* m,int r,int c) {
    mask_kernel<<<nsos::gpu::ceil_div_positive(static_cast<size_t>(r)*c,256),256,0,nsos::gpu::current_stream()>>>(x,m,r,c);check();
}
extern "C" void launch_cce_statistics(const float* z,float* s,const float* m,int r,int c,int v) {
    statistics_kernel<<<nsos::gpu::ceil_div_positive(r,128),128,0,nsos::gpu::current_stream()>>>(z,s,m,r,c,v); check();
}
extern "C" void launch_cce_finish(float* s,float* l,const float* m,int r,int v,float u,float b) {
    finish_kernel<<<nsos::gpu::ceil_div_positive(r,128),128,0,nsos::gpu::current_stream()>>>(s,l,m,r,v,u,b);check();
}
extern "C" void launch_cce_gradient(const float* z,float* g,const float* s,const float* m,int r,int c,int a,int v,float,float b,float k) {
    gradient_kernel<<<nsos::gpu::ceil_div_positive(static_cast<size_t>(r)*c,256),256,0,nsos::gpu::current_stream()>>>(z,g,s,m,r,c,a,v,b,k);check();
}
extern "C" void launch_cce_add_region(float* d,const float* s,int r,int c,int stride,int a,int b) {
    add_kernel<<<nsos::gpu::ceil_div_positive(static_cast<size_t>(r)*c,256),256,0,nsos::gpu::current_stream()>>>(d,s,r,c,stride,a,b);check();
}
