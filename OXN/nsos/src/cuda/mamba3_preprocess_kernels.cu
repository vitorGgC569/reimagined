#include "gpu_backend.h"
#include "cuda/mamba3_preprocess_kernels.cuh"
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nsos::mamba3_preprocess_gpu {
namespace {
constexpr int Threads=256,NormThreads=64,Tile=128;
struct Range {const void* p;std::size_t n;};
bool disjoint(const Range* reads,int nr,const Range* writes,int nw) {
    auto good=[](Range a) {const auto p=reinterpret_cast<std::uintptr_t>(a.p);return p&&p%4==0&&a.n&&p<=std::numeric_limits<std::uintptr_t>::max()-a.n;};
    auto overlap=[](Range a,Range b) {const auto p=reinterpret_cast<std::uintptr_t>(a.p),q=reinterpret_cast<std::uintptr_t>(b.p);return p<q+b.n&&q<p+a.n;};
    for (int i=0;i<nr;++i) if (!good(reads[i])) return false;
    for (int i=0;i<nw;++i) {
        if (!good(writes[i])) return false;
        for (int j=0;j<nr;++j) if (overlap(writes[i],reads[j])) return false;
        for (int j=i+1;j<nw;++j) if (overlap(writes[i],writes[j])) return false;
    }
    return true;
}
int blocks(std::size_t n,int threads=Threads) {
    if (!n||threads<=0) return 0;
    return static_cast<int>((std::min)(std::size_t(4096),(n-1)/static_cast<std::size_t>(threads)+1));
}
#if defined(NSOS_GPU_BACKEND_HIP)
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__>0)
#error "Mamba3 preprocessing requires finite checks and precise transcendental math"
#endif
__device__ float heavy(float x) {return x>=0?1+x:1/(1-x);}
__device__ float sigmoid(float x) {const float e=expf(-fabsf(x));return x>=0?1/(1+e):e/(1+e);}
__device__ float softplus(float x) {return fmaxf(x,0)+log1pf(expf(-fabsf(x)));}
__device__ bool bounded(float x) {return isfinite(x)&&fabsf(x)<=64;}
__device__ int active(Shape s,Input x,std::size_t row) {
    return static_cast<int>(row%s.sequence)<x.valid[row/s.sequence];
}
template<bool Reverse>
__global__ __launch_bounds__(Threads) void preflight(Shape s,Input x,Adjoint adj,int* status) {
    const int b=blockIdx.x,lane=threadIdx.x,length=x.valid[b];
    if constexpr (Reverse) {if (status[b]!=0) return;}
    if (length<0||length>s.sequence) {if (lane==0) status[b]=1;return;}
    __shared__ int flag[Threads];int bad=0;
    const auto row=static_cast<std::size_t>(b)*s.sequence;
    const auto nq=static_cast<std::size_t>(length)*s.groups*s.state_dim;
    for (std::size_t i=lane;i<nq;i+=Threads) {
        const auto at=row*s.groups*s.state_dim+i;
        if constexpr (Reverse) bad|=!isfinite(adj.q[at])||!isfinite(adj.k[at]);
        else bad|=!bounded(x.q[at])||!bounded(x.k[at]);
    }
    for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads;i+=Threads) {
        const auto at=row*s.heads+i;
        if constexpr (Reverse) bad|=!isfinite(adj.adt[at])||!isfinite(adj.dt[at]);
        else bad|=!bounded(x.raw_a[at])||!bounded(x.raw_dt[at]);
    }
    if constexpr (Reverse) {
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.groups*2;i+=Threads) {
            const float inv=x.inverse[row*s.groups*2+i];bad|=!isfinite(inv)||inv<=0;
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads*s.rotary_pairs;i+=Threads)
            bad|=!isfinite(adj.angles[row*s.heads*s.rotary_pairs+i]);
    } else {
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.rotary_pairs;i+=Threads)
            bad|=!bounded(x.angles[row*s.rotary_pairs+i]);
        if (length) {
            for (int n=lane;n<s.state_dim;n+=Threads) bad|=!bounded(x.q_norm[n])||!bounded(x.k_norm[n]);
            for (int h=lane;h<s.heads;h+=Threads) bad|=!bounded(x.dt_bias[h]);
        }
    }
    flag[lane]=bad;__syncthreads();
    if (lane==0) {for (int i=0;i<Threads;++i) bad|=flag[i];status[b]=bad?2:0;}
}
template<bool Reverse>
__global__ __launch_bounds__(NormThreads) void norm(Shape s,Config c,Input x,Prepared out,Adjoint adj,Gradient dx,const int* status) {
    // The RMS VJP subtracts a radial component. FP32 cached inverses and dots
    // lose the small epsilon residual for large, nearly radial adjoints.
    // Recompute only this reduction in FP64; inputs/outputs remain FP32.
    using Acc=std::conditional_t<Reverse,double,float>;
    __shared__ Acc square_q[NormThreads],square_k[NormThreads],cross_q[NormThreads],cross_k[NormThreads];
    const auto total=static_cast<std::size_t>(s.batch)*s.sequence*s.groups;
    const int n=threadIdx.x;
    for (std::size_t row=blockIdx.x;row<total;row+=gridDim.x) {
        const auto token=row/s.groups;const bool valid=status[token/s.sequence]==0&&active(s,x,token);
        const auto at=row*s.state_dim+n;
        const float q=valid&&n<s.state_dim?x.q[at]:0,k=valid&&n<s.state_dim?x.k[at]:0;
        if constexpr (Reverse) {
            cross_q[n]=valid&&n<s.state_dim?static_cast<double>(q)*x.q_norm[n]*adj.q[at]:0;
            cross_k[n]=valid&&n<s.state_dim?static_cast<double>(k)*x.k_norm[n]*adj.k[at]:0;
        }
        square_q[n]=static_cast<Acc>(q)*q;square_k[n]=static_cast<Acc>(k)*k;
        __syncthreads();
        for (int step=NormThreads/2;step;step/=2) {
            if (n<step) {
                if constexpr (Reverse) {cross_q[n]+=cross_q[n+step];cross_k[n]+=cross_k[n+step];}
                square_q[n]+=square_q[n+step];square_k[n]+=square_k[n+step];
            }
            __syncthreads();
        }
        if (n<s.state_dim) {
            if constexpr (Reverse) {
                const double sq=square_q[0]+static_cast<double>(s.state_dim)*c.norm_eps;
                const double sk=square_k[0]+static_cast<double>(s.state_dim)*c.norm_eps;
                dx.q[at]=valid?static_cast<float>(sqrt(static_cast<double>(s.state_dim)/sq)*
                    fma(-static_cast<double>(q),cross_q[0]/sq,static_cast<double>(x.q_norm[n])*adj.q[at])):0;
                dx.k[at]=valid?static_cast<float>(sqrt(static_cast<double>(s.state_dim)/sk)*
                    fma(-static_cast<double>(k),cross_k[0]/sk,static_cast<double>(x.k_norm[n])*adj.k[at])):0;
            } else {
                const float iq=1/sqrtf(square_q[0]/s.state_dim+c.norm_eps);
                const float ik=1/sqrtf(square_k[0]/s.state_dim+c.norm_eps);
                out.q[at]=valid?q*iq*x.q_norm[n]:0;out.k[at]=valid?k*ik*x.k_norm[n]:0;
            }
        }
        if constexpr (!Reverse) {
            if (n==0) {
                out.inverse[row*2]=valid?1/sqrtf(square_q[0]/s.state_dim+c.norm_eps):0;
                out.inverse[row*2+1]=valid?1/sqrtf(square_k[0]/s.state_dim+c.norm_eps):0;
            }
        }
        __syncthreads();
    }
}
template<bool Reverse>
__global__ __launch_bounds__(Threads) void time_angles(Shape s,Config c,Input x,Prepared out,Adjoint adj,Gradient dx,const int* status) {
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence;
    for (std::size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        const bool valid=status[row/s.sequence]==0&&active(s,x,row);
        for (int h=threadIdx.x;h<s.heads;h+=Threads) {
            const auto at=row*s.heads+h;
            if (valid) {
                const float act=heavy(x.raw_a[at]),a=-fmaxf(act,c.a_floor);
                const float raw=x.raw_dt[at]+x.dt_bias[h],dt=softplus(raw);
                if constexpr (Reverse) {
                    dx.raw_a[at]=act>c.a_floor?-adj.adt[at]*dt*(x.raw_a[at]>=0?1:act*act):0;
                    dx.raw_dt[at]=(adj.dt[at]+a*adj.adt[at])*sigmoid(raw);
                } else {out.dt[at]=dt;out.adt[at]=a*dt;}
            } else if constexpr (Reverse) {dx.raw_a[at]=0;dx.raw_dt[at]=0;}
            else {out.dt[at]=0;out.adt[at]=0;}
        }
        if constexpr (Reverse) {
            for (int r=threadIdx.x;r<s.rotary_pairs;r+=Threads) {
                float value=0;
                if (valid) for (int h=0;h<s.heads;++h) value+=adj.angles[(row*s.heads+h)*s.rotary_pairs+r];
                dx.angles[row*s.rotary_pairs+r]=value;
            }
        } else {
            for (int i=threadIdx.x;i<s.heads*s.rotary_pairs;i+=Threads)
                out.angles[row*s.heads*s.rotary_pairs+i]=valid?x.angles[row*s.rotary_pairs+i%s.rotary_pairs]:0;
        }
    }
}
__global__ __launch_bounds__(Threads) void norm_partials(Shape s,Config c,Input x,Adjoint adj,float* partial,const int* status) {
    __shared__ float acc[2*Threads];
    const int lane=threadIdx.x,n=lane%NormThreads,owner=lane/NormThreads;
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence*s.groups;
    // Four row owners per CTA, coalesced N, saved inverses. No row-by-row
    // square recomputation or barriers inside the tile; only two reductions.
    for (std::size_t part=blockIdx.x;part<(rows+Tile-1)/Tile;part+=gridDim.x) {
        float aq=0,ak=0;
        for (int offset=owner;offset<Tile;offset+=4) {
            const auto row=part*Tile+offset,token=row/s.groups;
            const bool valid=row<rows&&status[token/s.sequence]==0&&active(s,x,token);
            if (valid&&n<s.state_dim) {
                const auto at=row*s.state_dim+n;
                aq+=adj.q[at]*x.q[at]*x.inverse[row*2];
                ak+=adj.k[at]*x.k[at]*x.inverse[row*2+1];
            }
        }
        acc[lane]=aq;acc[Threads+lane]=ak;__syncthreads();
        for (int step=Threads/2;step>=NormThreads;step/=2) {
            if (lane<step) {acc[lane]+=acc[lane+step];acc[Threads+lane]+=acc[Threads+lane+step];}
            __syncthreads();
        }
        if (lane<s.state_dim) {partial[part*2*s.state_dim+lane]=acc[lane];partial[part*2*s.state_dim+s.state_dim+lane]=acc[Threads+lane];}
        __syncthreads();
    }
}
__global__ __launch_bounds__(Threads) void dt_partials(Shape s,Input x,Gradient dx,float* partial,const int* status) {
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence,parts=(rows+Tile-1)/Tile;
    float* dt=partial+((rows*s.groups+Tile-1)/Tile)*2*s.state_dim;
    for (std::size_t part=blockIdx.x;part<parts;part+=gridDim.x) {
        for (int h=threadIdx.x;h<s.heads;h+=Threads) {
            float value=0;
            for (int offset=0;offset<Tile;++offset) {
                const auto row=part*Tile+offset;
                if (row<rows&&status[row/s.sequence]==0&&active(s,x,row)) value+=dx.raw_dt[row*s.heads+h];
            }
            dt[part*s.heads+h]=value;
        }
    }
}
__global__ __launch_bounds__(Threads) void reduce_parameters(Shape s,Gradient dx,const float* partial) {
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence;
    const auto np=(rows*s.groups+Tile-1)/Tile,dp=(rows+Tile-1)/Tile;
    for (int n=blockIdx.x*Threads+threadIdx.x;n<s.state_dim;n+=gridDim.x*Threads) {
        float q=0,k=0;for (std::size_t part=0;part<np;++part) {q+=partial[part*2*s.state_dim+n];k+=partial[part*2*s.state_dim+s.state_dim+n];}
        dx.q_norm[n]=q;dx.k_norm[n]=k;
    }
    const float* dt=partial+np*2*s.state_dim;
    for (int h=blockIdx.x*Threads+threadIdx.x;h<s.heads;h+=gridDim.x*Threads) {
        float value=0;for (std::size_t part=0;part<dp;++part) value+=dt[part*s.heads+h];dx.dt_bias[h]=value;
    }
}
template<bool Reverse>
__global__ __launch_bounds__(Threads) void validate(Shape s,Prepared out,Gradient dx,int* status) {
    const int b=blockIdx.x,lane=threadIdx.x;
    if (status[b]!=0) return;
    const auto row=static_cast<std::size_t>(b)*s.sequence;
    __shared__ int range[Threads],numeric[Threads];int r=0,n=0;
    for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*s.state_dim;i+=Threads) {
        const auto at=row*s.groups*s.state_dim+i;
        const float q=Reverse?dx.q[at]:out.q[at],k=Reverse?dx.k[at]:out.k[at];
        n|=!isfinite(q)||!isfinite(k);
        if constexpr (!Reverse) r|=fabsf(q)>64||fabsf(k)>64;
    }
    for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads;i+=Threads) {
        const auto at=row*s.heads+i;
        const float a=Reverse?dx.raw_a[at]:out.adt[at],d=Reverse?dx.raw_dt[at]:out.dt[at];
        n|=!isfinite(a)||!isfinite(d);
        if constexpr (!Reverse) r|=a>0||d<0||d>16;
    }
    const auto angle_width=Reverse?s.rotary_pairs:s.heads*s.rotary_pairs;
    for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*angle_width;i+=Threads)
        n|=!isfinite(Reverse?dx.angles[row*angle_width+i]:out.angles[row*angle_width+i]);
    if constexpr (Reverse) {
        for (int i=lane;i<s.state_dim;i+=Threads) n|=!isfinite(dx.q_norm[i])||!isfinite(dx.k_norm[i]);
        for (int h=lane;h<s.heads;h+=Threads) n|=!isfinite(dx.dt_bias[h]);
    } else {
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*2;i+=Threads)
            n|=!isfinite(out.inverse[row*s.groups*2+i]);
    }
    range[lane]=r;numeric[lane]=n;__syncthreads();
    if (lane==0) {for (int i=0;i<Threads;++i) {r|=range[i];n|=numeric[i];}if (n||r) status[b]=n?3:2;}
}
template<bool Reverse>
__global__ __launch_bounds__(Threads) void clear_failed(Shape s,Prepared out,Gradient dx,const int* status) {
    const int b=blockIdx.x,lane=threadIdx.x;const auto row=static_cast<std::size_t>(b)*s.sequence;
    if (status[b]!=0) {
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*s.state_dim;i+=Threads) {
            const auto at=row*s.groups*s.state_dim+i;
            if constexpr (Reverse) {dx.q[at]=0;dx.k[at]=0;}else {out.q[at]=0;out.k[at]=0;}
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads;i+=Threads) {
            const auto at=row*s.heads+i;
            if constexpr (Reverse) {dx.raw_a[at]=0;dx.raw_dt[at]=0;}else {out.adt[at]=0;out.dt[at]=0;}
        }
        const auto width=Reverse?s.rotary_pairs:s.heads*s.rotary_pairs;
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*width;i+=Threads)
            if constexpr (Reverse) dx.angles[row*width+i]=0;else out.angles[row*width+i]=0;
        if constexpr (!Reverse) {
            for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*2;i+=Threads) out.inverse[row*s.groups*2+i]=0;
        }
    }
    if constexpr (Reverse) {
        if (b==0) {
            bool bad=false;for (int i=0;i<s.batch;++i) bad|=status[i]!=0;
            if (bad) {
                for (int n=lane;n<s.state_dim;n+=Threads) {dx.q_norm[n]=0;dx.k_norm[n]=0;}
                for (int h=lane;h<s.heads;h+=Threads) dx.dt_bias[h]=0;
            }
        }
    }
}
__global__ __launch_bounds__(Threads) void gate(Shape s,const int* status,mamba3_siso_gpu::Gradient dx,mamba3_siso_gpu::State di) {
    const int b=blockIdx.x,lane=threadIdx.x;const auto row=static_cast<std::size_t>(b)*s.sequence,bh=static_cast<std::size_t>(b)*s.heads;
    if (status[b]!=0) {
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.head_dim;i+=Threads) {
            dx.v[row*s.heads*s.head_dim+i]=0;if (dx.z) dx.z[row*s.heads*s.head_dim+i]=0;
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads;i+=Threads) dx.trap[row*s.heads+i]=0;
        for (int i=lane;i<s.heads*s.head_dim*s.state_dim;i+=Threads) di.ssm[bh*s.head_dim*s.state_dim+i]=0;
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) di.k[bh*s.state_dim+i]=0;
        for (int i=lane;i<s.heads*s.head_dim;i+=Threads) di.v[bh*s.head_dim+i]=0;
        for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) di.phase[bh*s.rotary_pairs+i]=0;
    }
    if (b==0) {
        bool bad=false;for (int i=0;i<s.batch;++i) bad|=status[i]!=0;
        if (bad) {
            for (int i=lane;i<s.heads*s.state_dim;i+=Threads) {dx.q_bias[i]=0;dx.k_bias[i]=0;}
            if (dx.d) for (int h=lane;h<s.heads;h+=Threads) dx.d[h]=0;
        }
    }
}
#endif
}
bool forward(Shape s,Config c,Input x,Prepared out,int* status) {
    if (!eligible(s,c)||!mamba3_siso_gpu::supported(s)) return false;
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence,q=rows*s.groups*s.state_dim*4,th=rows*s.heads*4,angles=rows*s.rotary_pairs*4;
    const Range reads[]={{x.q,q},{x.k,q},{x.raw_a,th},{x.raw_dt,th},{x.angles,angles},{x.q_norm,s.state_dim*4ull},{x.k_norm,s.state_dim*4ull},{x.dt_bias,s.heads*4ull},{x.valid,s.batch*4ull}};
    const Range writes[]={{out.q,q},{out.k,q},{out.adt,th},{out.dt,th},{out.angles,angles*s.heads},{status,s.batch*4ull}};
    const Range extra[]={{out.inverse,rows*s.groups*2*4}};
    // Include the inverse cache as a write in the full alias contract.
    const Range all_writes[]={writes[0],writes[1],writes[2],writes[3],writes[4],writes[5],extra[0]};
    if (!disjoint(reads,9,all_writes,7)) return false;
#if defined(NSOS_GPU_BACKEND_HIP)
    const auto stream=gpu::current_stream();
    preflight<false><<<s.batch,Threads,0,stream>>>(s,x,{},status);if (cudaGetLastError()!=cudaSuccess) return false;
    norm<false><<<blocks(rows*s.groups,1),NormThreads,0,stream>>>(s,c,x,out,{},{},status);if (cudaGetLastError()!=cudaSuccess) return false;
    time_angles<false><<<blocks(rows,1),Threads,0,stream>>>(s,c,x,out,{},{},status);if (cudaGetLastError()!=cudaSuccess) return false;
    validate<false><<<s.batch,Threads,0,stream>>>(s,out,{},status);if (cudaGetLastError()!=cudaSuccess) return false;
    clear_failed<false><<<s.batch,Threads,0,stream>>>(s,out,{},status);return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}
bool backward(Shape s,Config c,Input x,Adjoint adj,Gradient dx,float* partial,int* status) {
    if (!eligible(s,c)||!mamba3_siso_gpu::supported(s)) return false;
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence,q=rows*s.groups*s.state_dim*4,th=rows*s.heads*4,angles=rows*s.rotary_pairs*4;
    const Range reads[]={{x.q,q},{x.k,q},{x.raw_a,th},{x.raw_dt,th},{x.angles,angles},{x.q_norm,s.state_dim*4ull},{x.k_norm,s.state_dim*4ull},{x.dt_bias,s.heads*4ull},{x.valid,s.batch*4ull},
        {adj.q,q},{adj.k,q},{adj.adt,th},{adj.dt,th},{adj.angles,angles*s.heads},{x.inverse,rows*s.groups*2*4}};
    const Range writes[]={{dx.q,q},{dx.k,q},{dx.raw_a,th},{dx.raw_dt,th},{dx.angles,angles},{dx.q_norm,s.state_dim*4ull},{dx.k_norm,s.state_dim*4ull},{dx.dt_bias,s.heads*4ull},
        {partial,partial_elements(s)*4},{status,s.batch*4ull}};
    if (!disjoint(reads,15,writes,10)) return false;
#if defined(NSOS_GPU_BACKEND_HIP)
    const auto stream=gpu::current_stream();
    preflight<true><<<s.batch,Threads,0,stream>>>(s,x,adj,status);if (cudaGetLastError()!=cudaSuccess) return false;
    norm<true><<<blocks(rows*s.groups,1),NormThreads,0,stream>>>(s,c,x,{},adj,dx,status);if (cudaGetLastError()!=cudaSuccess) return false;
    time_angles<true><<<blocks(rows,1),Threads,0,stream>>>(s,c,x,{},adj,dx,status);if (cudaGetLastError()!=cudaSuccess) return false;
    norm_partials<<<blocks(partitions(s),1),Threads,0,stream>>>(s,c,x,adj,partial,status);if (cudaGetLastError()!=cudaSuccess) return false;
    dt_partials<<<blocks((rows+Tile-1)/Tile,1),Threads,0,stream>>>(s,x,dx,partial,status);if (cudaGetLastError()!=cudaSuccess) return false;
    reduce_parameters<<<blocks((std::max)(s.heads,s.state_dim)),Threads,0,stream>>>(s,dx,partial);if (cudaGetLastError()!=cudaSuccess) return false;
    validate<true><<<s.batch,Threads,0,stream>>>(s,{},dx,status);if (cudaGetLastError()!=cudaSuccess) return false;
    clear_failed<true><<<s.batch,Threads,0,stream>>>(s,{},dx,status);return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}
bool gate_chain(Shape s,const int* status,Gradient raw,mamba3_siso_gpu::Gradient rec,mamba3_siso_gpu::State di) {
    if (!mamba3_siso_gpu::supported(s)||!status||!rec.v||!rec.trap||!rec.q_bias||!rec.k_bias||!di.phase||!di.ssm||!di.k||!di.v||
        !raw.q||!raw.k||!raw.raw_a||!raw.raw_dt||!raw.angles||!raw.q_norm||!raw.k_norm||!raw.dt_bias) return false;
    const auto rows=static_cast<std::size_t>(s.batch)*s.sequence,bh=static_cast<std::size_t>(s.batch)*s.heads;
    Range writes[18];int count=0;
    auto put=[&](float* p,std::size_t n) {if (p) writes[count++]={p,n*4};};
    put(raw.q,rows*s.groups*s.state_dim);put(raw.k,rows*s.groups*s.state_dim);
    put(raw.raw_a,rows*s.heads);put(raw.raw_dt,rows*s.heads);put(raw.angles,rows*s.rotary_pairs);
    put(raw.q_norm,s.state_dim);put(raw.k_norm,s.state_dim);put(raw.dt_bias,s.heads);
    put(rec.v,rows*s.heads*s.head_dim);put(rec.z,rows*s.heads*s.head_dim);put(rec.trap,rows*s.heads);
    put(rec.q_bias,s.heads*s.state_dim);put(rec.k_bias,s.heads*s.state_dim);put(rec.d,s.heads);
    put(di.phase,bh*s.rotary_pairs);put(di.ssm,bh*s.head_dim*s.state_dim);put(di.k,bh*s.state_dim);put(di.v,bh*s.head_dim);
    const Range read[]={{status,s.batch*4ull}};
    if (!disjoint(read,1,writes,count)) return false;
#if defined(NSOS_GPU_BACKEND_HIP)
    const auto stream=gpu::current_stream();clear_failed<true><<<s.batch,Threads,0,stream>>>(s,{},raw,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    gate<<<s.batch,Threads,0,stream>>>(s,status,rec,di);return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}
}
