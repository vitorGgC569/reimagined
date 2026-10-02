#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/muon_optimizer.cuh"
#include "muon_math.h"
#include <cmath>
#include <limits>

namespace {
constexpr int T=16;
__device__ bool active(NsosActivityAwareOptimizerDesc d, int t) {
    return (!d.contribution.abort_issue || !*d.contribution.abort_issue) &&
        (!d.contribution.predicates || !d.contribution.predicates[t] || *d.contribution.predicates[t]);
}
__device__ float gradient_scale(const float* sqsum, float scale, float max_norm, int* issue) {
    if(!sqsum) return 1.0f;
    const float s=*sqsum, norm=scale*sqrtf(fmaxf(s,0.0f));
    if(!isfinite(s) || s<0 || !isfinite(norm)) {atomicExch(issue,2); return 0;}
    return scale*(norm>max_norm ? max_norm/(norm+1e-6f) : 1.0f);
}
__global__ void prepare(NsosActivityAwareOptimizerDesc d,int t,int rows,int cols,
    float* x,double* norm,const float* sqsum,float scale,float max_norm,int* issue) {
    if(!active(d,t)) return;
    __shared__ double partial[256];
    const float gs=gradient_scale(sqsum,scale,max_norm,issue);
    const bool transpose=rows>cols; const int c=transpose?rows:cols;
    double sum=0;
    for(size_t i=threadIdx.x;i<size_t(rows)*cols;i+=blockDim.x) {
        const float g=d.g[t][i]*gs;
        const float m=d.m[t][i]*nsos::muon::kMomentum+g*(1-nsos::muon::kMomentum);
        d.m[t][i]=m;
        const float u=g*(1-nsos::muon::kMomentum)+m*nsos::muon::kMomentum;
        const size_t dest=transpose ? (i%cols)*c+i/cols : i;
        x[dest]=u;sum+=double(u)*u;
        if(!isfinite(m)||!isfinite(u))atomicExch(issue,2);
    }
    partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) {*norm=partial[0];if(!isfinite(partial[0]))atomicExch(issue,2);}
}
__global__ void normalize(NsosActivityAwareOptimizerDesc d,int t,float* x,size_t n,const double* norm,int* issue) {
    if(!active(d,t))return;
    const float denominator=float(sqrt(*norm))+nsos::muon::kEpsilon;
    if(!isfinite(denominator))atomicExch(issue,2);
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)x[i]/=denominator;
}
// All three GEMMs use bounded 16x16 LDS tiles and FP32 accumulators. Uniform
// activity checks precede shared-memory barriers; inactive scratch is untouched.
// kind 0: X X^T; kind 1: b*A+c*A*A; kind 2: a*X+B*X.
template<int Kind>
__global__ void multiply(NsosActivityAwareOptimizerDesc d,int t,const float* left,
    const float* right,float* out,int r,int c) {
    if(!active(d,t))return;
    __shared__ float l[T][T], q[T][T];
    const int row=int(blockIdx.y)*T+threadIdx.y, col=int(blockIdx.x)*T+threadIdx.x;
    const int width=Kind==2?c:r, inner=Kind==0?c:r;
    float acc=0;
    for(int base=0;base<inner;base+=T) {
        const int lk=base+threadIdx.x, rk=base+threadIdx.y;
        l[threadIdx.y][threadIdx.x]=row<r && lk<inner ? left[size_t(row)*inner+lk] : 0;
        if constexpr(Kind==0)q[threadIdx.y][threadIdx.x]=col<r && rk<c ? right[size_t(col)*c+rk] : 0;
        else q[threadIdx.y][threadIdx.x]=rk<r && col<width ? right[size_t(rk)*width+col] : 0;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<T;++k)acc+=l[threadIdx.y][k]*q[k][threadIdx.x];
        __syncthreads();
    }
    if(row<r && col<width) {
        if constexpr(Kind==1)acc=nsos::muon::kB*left[size_t(row)*r+col]+nsos::muon::kC*acc;
        if constexpr(Kind==2)acc=nsos::muon::kA*right[size_t(row)*c+col]+acc;
        out[size_t(row)*width+col]=acc;
    }
}
__global__ void finish_direction(NsosActivityAwareOptimizerDesc d,int t,const float* x,
    float* result,int rows,int cols,int* issue) {
    if(!active(d,t))return;
    const bool transpose=rows>cols; const int c=transpose?rows:cols;
    const float adjustment=sqrtf(fmaxf(1.0f,float(rows)/cols));
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(rows)*cols;i+=size_t(gridDim.x)*blockDim.x) {
        const float u=x[transpose ? (i%cols)*c+i/cols : i]*adjustment;
        result[i]=u;if(!isfinite(u))atomicExch(issue,2);
    }
}
__global__ void epilogue(NsosActivityAwareOptimizerDesc d,const unsigned char* muon_mask,
    float* const* directions,const float* sqsum,float scale,float max_norm,float b1,float b2,
    float bc1,float bc2,float eps,float wd,bool clear,int* issue) {
    // Static tensor descriptors, no host read of the activity cohort. Multiple
    // blocks cooperate per tensor, preventing a large head from owning one CU.
    const int t=blockIdx.y;
    if(!active(d,t))return;
    const bool muon=muon_mask[t]!=0;
    const float gs=gradient_scale(sqsum,scale,max_norm,issue), lr=d.learning_rates[t];
    const size_t n=d.offsets[t+1]-d.offsets[t];
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        float delta;
        float m_hat=0.0f, denominator=1.0f;
        if(muon)delta=directions[t][i];
        else {
            const float g=d.g[t][i]*gs;
            const float m=b1*d.m[t][i]+(1-b1)*g;
            const float v=b2*d.v[t][i]+(1-b2)*g*g;
            d.m[t][i]=m;d.v[t][i]=v;
            m_hat=m/bc1;denominator=sqrtf(v/bc2)+eps;
            delta=m_hat/denominator;
            if(!isfinite(m)||!isfinite(v))atomicExch(issue,2);
        }
        float w=d.w[t][i];if(d.wd_flags[t])w-=lr*wd*w;
        // Preserve the serial AdamW FP32 expression tree. Multiplying lr
        // before division avoids a different rounding/FMA trajectory that
        // can cross lowp operand conversion thresholds on later VJPs.
        // The existing gfx1102 Muon update contracts this multiply/subtract
        // into FMA. Keep that arithmetic when splitting the Adam branch.
        if(muon)w=fmaf(-lr,delta,w);
        else w-=lr*m_hat/denominator;
        d.w[t][i]=w;
        if(clear)d.g[t][i]=0;
        if(!isfinite(delta)||!isfinite(w))atomicExch(issue,2);
    }
}
bool accepted(){return cudaGetLastError()==cudaSuccess;}
bool valid(NsosActivityAwareOptimizerDesc d,int* issue) {
    return d.n_tensors>0 && d.n_tensors<=65535 && d.total && d.w && d.g && d.m && d.v &&
        d.offsets && d.learning_rates && d.wd_flags && d.contribution.abort_issue &&
        issue && issue!=d.contribution.abort_issue;
}
}
extern "C" bool launch_activity_muon_direction(NsosActivityAwareOptimizerDesc d,
    int t,int rows,int cols,float* x,float* y,float* gram,float* poly,float* direction,
    double* norm,const float* sqsum,float scale,float max_norm,int* issue) {
    if(!valid(d,issue)||t<0||t>=d.n_tensors||rows<=0||cols<=0||!x||!y||!gram||!poly||!direction||!norm||
        !std::isfinite(scale)||scale<=0||!std::isfinite(max_norm)||max_norm<=0)return false;
    const int r=rows<cols?rows:cols, c=rows>cols?rows:cols;
    const auto stream=nsos::gpu::current_stream();
    const unsigned int blocks=static_cast<unsigned int>((std::min)(size_t(65535),(size_t(rows)*cols+255)/256));
    if(static_cast<unsigned int>(r)>65535u*T)return false;
    const unsigned int row_tiles=(static_cast<unsigned int>(r)+T-1)/T;
    const unsigned int col_tiles=(static_cast<unsigned int>(c)+T-1)/T;
    const dim3 tile(T,T), square(row_tiles,row_tiles), rectangular(col_tiles,row_tiles);
    prepare<<<1,256,0,stream>>>(d,t,rows,cols,x,norm,sqsum,scale,max_norm,issue);if(!accepted())return false;
    normalize<<<blocks,256,0,stream>>>(d,t,x,size_t(rows)*cols,norm,issue);if(!accepted())return false;
    for(int step=0;step<nsos::muon::kSteps;++step) {
        multiply<0><<<square,tile,0,stream>>>(d,t,x,x,gram,r,c);if(!accepted())return false;
        multiply<1><<<square,tile,0,stream>>>(d,t,gram,gram,poly,r,c);if(!accepted())return false;
        multiply<2><<<rectangular,tile,0,stream>>>(d,t,poly,x,y,r,c);if(!accepted())return false;
        auto* previous=x;x=y;y=previous;
    }
    finish_direction<<<blocks,256,0,stream>>>(d,t,x,direction,rows,cols,issue);return accepted();
}
extern "C" bool launch_activity_hybrid_muon_adam_epilogue(NsosActivityAwareOptimizerDesc d,
    const unsigned char* muon_mask,float* const* directions,const float* sqsum,float scale,
    float max_norm,float b1,float b2,float bc1,float bc2,float eps,float wd,bool clear,int* issue) {
    if(!valid(d,issue)||!muon_mask||!directions||!std::isfinite(scale)||scale<=0||
        !std::isfinite(max_norm)||max_norm<=0||!std::isfinite(b1)||b1<0||b1>=1||
        !std::isfinite(b2)||b2<0||b2>=1||!std::isfinite(bc1)||bc1<=0||bc1>1||
        !std::isfinite(bc2)||bc2<=0||bc2>1||!std::isfinite(eps)||eps<=0||!std::isfinite(wd)||wd<0)return false;
    epilogue<<<dim3(32,d.n_tensors),256,0,nsos::gpu::current_stream()>>>(d,muon_mask,directions,
        sqsum,scale,max_norm,b1,b2,bc1,bc2,eps,wd,clear,issue);return accepted();
}
