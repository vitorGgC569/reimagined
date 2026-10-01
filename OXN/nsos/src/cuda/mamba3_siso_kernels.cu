#include "gpu_backend.h"
#include "cuda/mamba3_siso_kernels.cuh"
#include <cstdint>
#include <initializer_list>
#include <string>

namespace nsos::mamba3_siso_gpu {
namespace {
constexpr int Threads=256,Nmax=64,Pmax=64,Cells=4096;
constexpr float Pi=3.14159265358979323846f,TwoPi=2*Pi;
struct Range {const void* data;std::size_t bytes;};
bool disjoint(std::initializer_list<Range> reads,std::initializer_list<Range> writes) {
    auto valid=[](Range r) {const auto p=reinterpret_cast<std::uintptr_t>(r.data);return r.data&&r.bytes&&p%4==0&&p<=std::numeric_limits<std::uintptr_t>::max()-r.bytes;};
    auto overlap=[](Range a,Range b) {const auto p=reinterpret_cast<std::uintptr_t>(a.data),q=reinterpret_cast<std::uintptr_t>(b.data);return p<q+b.bytes&&q<p+a.bytes;};
    for (auto r:reads) if (!valid(r)) return false;
    for (auto w:writes) if (!valid(w)) return false;
    for (auto a=writes.begin();a!=writes.end();++a) {
        for (auto b=a+1;b!=writes.end();++b) if (overlap(*a,*b)) return false;
        for (auto r:reads) if (overlap(*a,r)) return false;
    }
    return true;
}
bool empty(ConstState s) {return !s.phase&&!s.ssm&&!s.k&&!s.v;}
bool complete(ConstState s) {return s.phase&&s.ssm&&s.k&&s.v;}
ConstState as_const(State s) {return {s.phase,s.ssm,s.k,s.v};}
bool host_input(Input x) {return x.q&&x.k&&x.v&&x.adt&&x.dt&&x.trap&&x.angles&&x.q_bias&&x.k_bias&&x.valid;}

#if defined(NSOS_GPU_BACKEND_HIP)
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__>0)
#error "Mamba3 SISO requires finite checks and precise expf"
#endif
__device__ std::size_t time_head(Shape s,int b,int t,int h) {return (static_cast<std::size_t>(b)*s.sequence+t)*s.heads+h;}
__device__ float sigmoid(float x) {const float e=expf(-fabsf(x));return x>=0?1/(1+e):e/(1+e);}
__device__ bool bounded(float x,float bound=64) {return isfinite(x)&&fabsf(x)<=bound;}
__device__ void offsets(Shape s,int bh,int c,std::size_t& ph,std::size_t& hs,std::size_t& ks,std::size_t& vs) {
    const int count=1+(s.sequence-1)/s.chunk;
    const auto base=static_cast<std::size_t>(bh)*count*(s.rotary_pairs+s.head_dim*s.state_dim+s.state_dim+s.head_dim);
    ph=base+static_cast<std::size_t>(c)*s.rotary_pairs;
    hs=base+static_cast<std::size_t>(count)*s.rotary_pairs+static_cast<std::size_t>(c)*s.head_dim*s.state_dim;
    ks=base+static_cast<std::size_t>(count)*(s.rotary_pairs+s.head_dim*s.state_dim)+static_cast<std::size_t>(c)*s.state_dim;
    vs=base+static_cast<std::size_t>(count)*(s.rotary_pairs+s.head_dim*s.state_dim+s.state_dim)+static_cast<std::size_t>(c)*s.head_dim;
}
template<bool Reverse>
__global__ __launch_bounds__(Threads) void preflight(Shape s,Input x,ConstState initial,
    const float* dy,ConstState seed,int* status,const int* upstream=nullptr) {
    __shared__ int flags[Threads];
    const int b=blockIdx.x,lane=threadIdx.x,length=x.valid[b];
    if constexpr (Reverse) {if (status[b]!=0) return;}
    else if (upstream&&upstream[b]!=0) {
        if (lane==0) status[b]=upstream[b]>=1&&upstream[b]<=3?upstream[b]:2;
        return;
    }
    if (length<0||length>s.sequence) {if (lane==0) status[b]=1;return;}
    int bad=0;
    const auto head_begin=static_cast<std::size_t>(b)*s.sequence*s.heads;
    if constexpr (!Reverse) {
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.groups*s.state_dim;i+=Threads) {
            const auto index=static_cast<std::size_t>(b)*s.sequence*s.groups*s.state_dim+i;
            bad|=!bounded(x.q[index])||!bounded(x.k[index]);
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads;i+=Threads) {
            const auto th=head_begin+i;
            bad|=!isfinite(x.adt[th])||x.adt[th]>0||!isfinite(x.dt[th])||x.dt[th]<0||x.dt[th]>16||!bounded(x.trap[th]);
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads*s.rotary_pairs;i+=Threads) bad|=!bounded(x.angles[head_begin*s.rotary_pairs+i]);
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads*s.head_dim;i+=Threads) {
            bad|=!bounded(x.v[head_begin*s.head_dim+i]);if (x.z) bad|=!bounded(x.z[head_begin*s.head_dim+i]);
        }
        if (length) for (int i=lane;i<s.heads*s.state_dim;i+=Threads) bad|=!bounded(x.q_bias[i])||!bounded(x.k_bias[i]);
        if (x.d&&length) for (int h=lane;h<s.heads;h+=Threads) bad|=!bounded(x.d[h]);
    } else {
        for (std::size_t i=lane;i<static_cast<std::size_t>(length)*s.heads*s.head_dim;i+=Threads) bad|=!isfinite(dy[head_begin*s.head_dim+i]);
    }
    ConstState value=Reverse?seed:initial;
    if (value.phase) {
        const auto bh=static_cast<std::size_t>(b)*s.heads;
        for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) {
            const float phase=value.phase[bh*s.rotary_pairs+i];
            if constexpr (Reverse) bad|=!isfinite(phase);else bad|=!bounded(phase,TwoPi);
        }
        for (int i=lane;i<s.heads*s.state_dim*s.head_dim;i+=Threads) bad|=!isfinite(value.ssm[bh*s.state_dim*s.head_dim+i]);
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) bad|=!isfinite(value.k[bh*s.state_dim+i]);
        for (int i=lane;i<s.heads*s.head_dim;i+=Threads) bad|=!isfinite(value.v[bh*s.head_dim+i]);
    }
    flags[lane]=bad;__syncthreads();
    if (lane==0) {int any=0;for (int i=0;i<Threads;++i) any|=flags[i];status[b]=any?2:0;}
}
// RoPE pairs are adjacent, matching original SISO (MIMO has a different layout).
__device__ void rotate(Shape s,Input x,int b,int t,int h,float* phase,float* q,float* k) {
    const auto th=time_head(s,b,t,h);
    for (int r=threadIdx.x;r<s.rotary_pairs;r+=Threads) {
        float p=phase[r]+Pi*tanhf(x.angles[th*s.rotary_pairs+r])*x.dt[th];
        phase[r]=p-TwoPi*floorf(p/TwoPi);
    }
    __syncthreads();
    for (int pair=threadIdx.x;pair<s.state_dim/2;pair+=Threads) {
        const int n=pair*2;const float angle=pair<s.rotary_pairs?phase[pair]:0;
        const float c=cosf(angle),sn=sinf(angle);
        const auto qi=((static_cast<std::size_t>(b)*s.sequence+t)*s.groups+h/(s.heads/s.groups))*s.state_dim+n;
        const auto bias=static_cast<std::size_t>(h)*s.state_dim+n;
        const float qa=x.q[qi]+x.q_bias[bias],qb=x.q[qi+1]+x.q_bias[bias+1];
        const float ka=x.k[qi]+x.k_bias[bias],kb=x.k[qi+1]+x.k_bias[bias+1];
        q[n]=qa*c-qb*sn;q[n+1]=qa*sn+qb*c;k[n]=ka*c-kb*sn;k[n+1]=ka*sn+kb*c;
    }
    __syncthreads();
}
__global__ __launch_bounds__(Threads) void scan(Shape s,Input x,ConstState initial,
    State final,float* out,float* boundaries,int* status) {
    __shared__ float hs[Cells],phase[Nmax/2],pk[Nmax],pv[Pmax],q[Nmax],k[Nmax];
    __shared__ int length;
    const int bh=blockIdx.x,b=bh/s.heads,h=bh%s.heads,lane=threadIdx.x;
    // Other head CTAs may latch a numerical failure concurrently. Exactly one
    // lane snapshots the loop bound; otherwise lanes could observe different
    // status values and diverge around the barriers inside the time loop.
    if (lane==0) length=atomicAdd(status+b,0)==0?x.valid[b]:0;
    __syncthreads();
    const auto base=static_cast<std::size_t>(bh);
    for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) hs[i]=initial.ssm?initial.ssm[base*s.head_dim*s.state_dim+i]:0;
    if (lane<s.rotary_pairs) phase[lane]=initial.phase?initial.phase[base*s.rotary_pairs+lane]:0;
    if (lane<s.state_dim) pk[lane]=initial.k?initial.k[base*s.state_dim+lane]:0;
    if (lane<s.head_dim) pv[lane]=initial.v?initial.v[base*s.head_dim+lane]:0;
    __syncthreads();
    for (int t=0;t<length;++t) {
        const auto th=time_head(s,b,t,h),vp=th*s.head_dim;
        if (t%s.chunk==0) {
            std::size_t ph,sh,kh,vh;offsets(s,bh,t/s.chunk,ph,sh,kh,vh);
            for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) boundaries[sh+i]=hs[i];
            if (lane<s.rotary_pairs) boundaries[ph+lane]=phase[lane];
            if (lane<s.state_dim) boundaries[kh+lane]=pk[lane];
            if (lane<s.head_dim) boundaries[vh+lane]=pv[lane];
        }
        rotate(s,x,b,t,h,phase,q,k);
        const float a=expf(x.adt[th]),lambda=sigmoid(x.trap[th]),beta=(1-lambda)*x.dt[th]*a,gamma=lambda*x.dt[th];
        for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) {
            const int p=i/s.state_dim,n=i%s.state_dim;
            hs[i]=a*hs[i]+beta*pk[n]*pv[p]+gamma*k[n]*x.v[vp+p];
        }
        __syncthreads();
        if (lane<s.head_dim) {
            float y=0;for (int n=0;n<s.state_dim;++n) y+=hs[lane*s.state_dim+n]*q[n];
            if (x.d) y+=x.d[h]*x.v[vp+lane];
            if (x.z) y*=x.z[vp+lane]*sigmoid(x.z[vp+lane]);
            out[vp+lane]=y;
            if (!isfinite(y)) atomicExch(status+b,3);
            pv[lane]=x.v[vp+lane];
        }
        if (lane<s.state_dim) pk[lane]=k[lane];
        __syncthreads();
    }
    // Error status is consumed only by later kernels, never a loop/barrier gate.
    // Numerical failures are cleared in a batch-owned finalize after this scan.
    for (int t=length;t<s.sequence;++t) if (lane<s.head_dim) out[time_head(s,b,t,h)*s.head_dim+lane]=0;
    for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) final.ssm[base*s.head_dim*s.state_dim+i]=hs[i];
    if (lane<s.rotary_pairs) final.phase[base*s.rotary_pairs+lane]=phase[lane];
    if (lane<s.state_dim) final.k[base*s.state_dim+lane]=pk[lane];
    if (lane<s.head_dim) final.v[base*s.head_dim+lane]=pv[lane];
}
__global__ __launch_bounds__(Threads) void finalize_forward(Shape s,State final,float* out,int* status) {
    const int b=blockIdx.x,lane=threadIdx.x;int bad=status[b];
    __shared__ int flags[Threads];int own=0;
    const auto bh=static_cast<std::size_t>(b)*s.heads;
    for (int i=lane;i<s.heads*s.head_dim*s.state_dim;i+=Threads) own|=!isfinite(final.ssm[bh*s.head_dim*s.state_dim+i]);
    for (int i=lane;i<s.heads*s.state_dim;i+=Threads) own|=!isfinite(final.k[bh*s.state_dim+i]);
    for (int i=lane;i<s.heads*s.head_dim;i+=Threads) own|=!isfinite(final.v[bh*s.head_dim+i]);
    for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) own|=!isfinite(final.phase[bh*s.rotary_pairs+i]);
    flags[lane]=own;__syncthreads();
    if (lane==0) {for (int i=0;i<Threads;++i) bad|=flags[i]?3:0;if (status[b]==0&&bad) status[b]=3;}
    __syncthreads();if (status[b]==0) return;
    for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.head_dim;i+=Threads) out[bh*s.sequence*s.head_dim+i]=0;
    for (int i=lane;i<s.heads*s.head_dim*s.state_dim;i+=Threads) final.ssm[bh*s.head_dim*s.state_dim+i]=0;
    for (int i=lane;i<s.heads*s.state_dim;i+=Threads) final.k[bh*s.state_dim+i]=0;
    for (int i=lane;i<s.heads*s.head_dim;i+=Threads) final.v[bh*s.head_dim+i]=0;
    for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) final.phase[bh*s.rotary_pairs+i]=0;
}

__global__ __launch_bounds__(Threads) void reverse_scan(Shape s,Input x,ConstState initial,
    const float* dy,ConstState seed,Gradient dx,State di,const float* boundaries,
    float* replay,float* partials,int* status) {
    __shared__ float adj[Cells],gh[Cells],phase[Nmax/2],gp[Nmax/2],pk[Nmax],pv[Pmax],
        q[Nmax],k[Nmax],gk[Nmax],gv[Pmax],dq[Nmax],dk[Nmax],gy[Pmax],
        qbgrad[Nmax],kbgrad[Nmax],phase_dt[Nmax/2],tree[3*Threads],dgrad;
    const int bh=blockIdx.x,b=bh/s.heads,h=bh%s.heads,lane=threadIdx.x;
    const auto base=static_cast<std::size_t>(bh);
    const int length=status[b]==0?x.valid[b]:0;
    const std::size_t hist_size=static_cast<std::size_t>(s.chunk+1)*s.head_dim*s.state_dim;
    const std::size_t replay_width=hist_size+2*s.chunk*s.state_dim+s.chunk*s.rotary_pairs;
    float* history=replay+base*replay_width;
    float* qr=history+hist_size;float* kr=qr+s.chunk*s.state_dim;float* phases=kr+s.chunk*s.state_dim;
    const std::size_t expanded=static_cast<std::size_t>(s.batch)*s.heads*s.sequence*s.state_dim;
    float* gq=partials;float* gkr=partials+expanded;
    float* qbg=partials+2*expanded;float* kbg=qbg+static_cast<std::size_t>(s.batch)*s.heads*s.state_dim;
    float* dg=kbg+static_cast<std::size_t>(s.batch)*s.heads*s.state_dim;
    const bool ok=status[b]==0;
    for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) adj[i]=ok&&seed.ssm?seed.ssm[base*s.head_dim*s.state_dim+i]:0;
    if (lane<s.rotary_pairs) gp[lane]=ok&&seed.phase?seed.phase[base*s.rotary_pairs+lane]:0;
    if (lane<s.state_dim) {gk[lane]=ok&&seed.k?seed.k[base*s.state_dim+lane]:0;qbgrad[lane]=0;kbgrad[lane]=0;}
    if (lane<s.head_dim) gv[lane]=ok&&seed.v?seed.v[base*s.head_dim+lane]:0;
    if (lane==0) dgrad=0;
    __syncthreads();
    // Each CTA owns its reverse carry for the WHOLE sequence. Scratch is only
    // one chunk; crossing a boundary never truncates phase/SSM/K/V adjoints.
    for (int c=(length-1)/s.chunk;length>0&&c>=0;--c) {
        const int start=c*s.chunk,count=min(s.chunk,length-start);
        std::size_t ph,sh,kh,vh;offsets(s,bh,c,ph,sh,kh,vh);
        for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) history[i]=boundaries[sh+i];
        if (lane<s.rotary_pairs) phase[lane]=boundaries[ph+lane];
        if (lane<s.state_dim) pk[lane]=boundaries[kh+lane];
        if (lane<s.head_dim) pv[lane]=boundaries[vh+lane];
        __syncthreads();
        for (int l=0;l<count;++l) {
            const int t=start+l;const auto th=time_head(s,b,t,h),vp=th*s.head_dim;
            rotate(s,x,b,t,h,phase,q,k);
            if (lane<s.rotary_pairs) phases[l*s.rotary_pairs+lane]=phase[lane];
            if (lane<s.state_dim) {qr[l*s.state_dim+lane]=q[lane];kr[l*s.state_dim+lane]=k[lane];}
            const float a=expf(x.adt[th]),lambda=sigmoid(x.trap[th]),beta=(1-lambda)*x.dt[th]*a,gamma=lambda*x.dt[th];
            for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) {
                const int p=i/s.state_dim,n=i%s.state_dim;
                history[(l+1)*s.head_dim*s.state_dim+i]=a*history[l*s.head_dim*s.state_dim+i]+beta*pk[n]*pv[p]+gamma*k[n]*x.v[vp+p];
            }
            __syncthreads();
            if (lane<s.state_dim) pk[lane]=k[lane];
            if (lane<s.head_dim) pv[lane]=x.v[vp+lane];
            __syncthreads();
        }
        for (int l=count-1;l>=0;--l) {
            const int t=start+l;const auto th=time_head(s,b,t,h),vp=th*s.head_dim;
            if (lane<s.state_dim) {q[lane]=qr[l*s.state_dim+lane];k[lane]=kr[l*s.state_dim+lane];pk[lane]=l?kr[(l-1)*s.state_dim+lane]:boundaries[kh+lane];}
            if (lane<s.head_dim) pv[lane]=t?x.v[time_head(s,b,t-1,h)*s.head_dim+lane]:(initial.v?initial.v[base*s.head_dim+lane]:0);
            if (lane<s.rotary_pairs) phase[lane]=phases[l*s.rotary_pairs+lane];
            __syncthreads();
            const float a=expf(x.adt[th]),lambda=sigmoid(x.trap[th]),beta=(1-lambda)*x.dt[th]*a,gamma=lambda*x.dt[th];
            if (lane<s.head_dim) {
                float y=0;for (int n=0;n<s.state_dim;++n) y+=history[(l+1)*s.head_dim*s.state_dim+lane*s.state_dim+n]*q[n];
                if (x.d) y+=x.d[h]*x.v[vp+lane];
                float g=dy[vp+lane];
                if (x.z) {const float sig=sigmoid(x.z[vp+lane]);dx.z[vp+lane]=g*y*(sig+x.z[vp+lane]*sig*(1-sig));g*=x.z[vp+lane]*sig;}
                gy[lane]=g;
            }
            __syncthreads();
            float da=0,db=0,dc=0;
            for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) {
                const int p=i/s.state_dim,n=i%s.state_dim;
                const float g=adj[i]+gy[p]*q[n];gh[i]=g;adj[i]=g*a;
                da+=g*history[l*s.head_dim*s.state_dim+i];db+=g*pk[n]*pv[p];dc+=g*k[n]*x.v[vp+p];
            }
            tree[lane]=da;tree[Threads+lane]=db;tree[2*Threads+lane]=dc;
            __syncthreads();
            for (int step=Threads/2;step>0;step/=2) {
                if (lane<step) {tree[lane]+=tree[lane+step];tree[Threads+lane]+=tree[Threads+lane+step];tree[2*Threads+lane]+=tree[2*Threads+lane+step];}
                __syncthreads();
            }
            if (lane==0) {
                dx.adt[th]=(tree[0]+tree[Threads]*(1-lambda)*x.dt[th])*a;
                dx.dt[th]=tree[Threads]*(1-lambda)*a+tree[2*Threads]*lambda;
                dx.trap[th]=(-tree[Threads]*x.dt[th]*a+tree[2*Threads]*x.dt[th])*lambda*(1-lambda);
                if (x.d) for (int p=0;p<s.head_dim;++p) dgrad+=gy[p]*x.v[vp+p];
            }
            if (lane<s.state_dim) {
                float gq0=0,gk0=gk[lane],prior=0;
                for (int p=0;p<s.head_dim;++p) {
                    gq0+=gy[p]*history[(l+1)*s.head_dim*s.state_dim+p*s.state_dim+lane];
                    gk0+=gh[p*s.state_dim+lane]*gamma*x.v[vp+p];prior+=gh[p*s.state_dim+lane]*beta*pv[p];
                }
                dq[lane]=gq0;dk[lane]=gk0;gk[lane]=prior;
            }
            if (lane<s.head_dim) {
                float g=gv[lane]+(x.d?gy[lane]*x.d[h]:0),prior=0;
                for (int n=0;n<s.state_dim;++n) {g+=gh[lane*s.state_dim+n]*gamma*k[n];prior+=gh[lane*s.state_dim+n]*beta*pk[n];}
                dx.v[vp+lane]=g;gv[lane]=prior;
            }
            __syncthreads();
            if (lane<s.state_dim/2) {
                const int n=lane*2;const float ang=lane<s.rotary_pairs?phase[lane]:0,c0=cosf(ang),sn=sinf(ang);
                const float q0=dq[n]*c0+dq[n+1]*sn,q1=-dq[n]*sn+dq[n+1]*c0;
                const float k0=dk[n]*c0+dk[n+1]*sn,k1=-dk[n]*sn+dk[n+1]*c0;
                const auto dst=(base*s.sequence+t)*s.state_dim+n;
                gq[dst]=q0;gq[dst+1]=q1;gkr[dst]=k0;gkr[dst+1]=k1;
                qbgrad[n]+=q0;qbgrad[n+1]+=q1;kbgrad[n]+=k0;kbgrad[n+1]+=k1;
                if (lane<s.rotary_pairs) {
                    gp[lane]+=-dq[n]*q[n+1]+dq[n+1]*q[n]-dk[n]*k[n+1]+dk[n+1]*k[n];
                    const float angle=tanhf(x.angles[th*s.rotary_pairs+lane]);
                    dx.angles[th*s.rotary_pairs+lane]=gp[lane]*Pi*(1-angle*angle)*x.dt[th];phase_dt[lane]=gp[lane]*Pi*angle;
                }
            }
            __syncthreads();
            if (lane==0) for (int r=0;r<s.rotary_pairs;++r) dx.dt[th]+=phase_dt[r];
            __syncthreads();
        }
    }
    for (int t=length;t<s.sequence;++t) {
        const auto th=time_head(s,b,t,h);
        if (lane<s.head_dim) {dx.v[th*s.head_dim+lane]=0;if (x.z) dx.z[th*s.head_dim+lane]=0;}
        if (lane<s.rotary_pairs) dx.angles[th*s.rotary_pairs+lane]=0;
        if (lane<s.state_dim) {gq[(base*s.sequence+t)*s.state_dim+lane]=0;gkr[(base*s.sequence+t)*s.state_dim+lane]=0;}
        if (lane==0) {dx.adt[th]=0;dx.dt[th]=0;dx.trap[th]=0;}
    }
    for (int i=lane;i<s.head_dim*s.state_dim;i+=Threads) di.ssm[base*s.head_dim*s.state_dim+i]=adj[i];
    if (lane<s.state_dim) {di.k[base*s.state_dim+lane]=gk[lane];qbg[base*s.state_dim+lane]=qbgrad[lane];kbg[base*s.state_dim+lane]=kbgrad[lane];}
    if (lane<s.head_dim) di.v[base*s.head_dim+lane]=gv[lane];
    if (lane<s.rotary_pairs) di.phase[base*s.rotary_pairs+lane]=gp[lane];
    if (lane==0) dg[base]=dgrad;
}
// Parameter/GQA reductions have one deterministic owner. No atomics and no
// hidden host scale/materialization; order is heads then batches.
__global__ __launch_bounds__(Threads) void reduce_gradients(Shape s,Gradient dx,const float* partials,const int* status) {
    const auto bh=static_cast<std::size_t>(s.batch)*s.heads;
    const auto expanded=bh*s.sequence*s.state_dim;
    const float* gq=partials;const float* gk=partials+expanded;
    const float* qb=partials+2*expanded;const float* kb=qb+bh*s.state_dim;const float* gd=kb+bh*s.state_dim;
    const auto total=static_cast<std::size_t>(s.batch)*s.sequence*s.groups*s.state_dim;
    const auto stride=static_cast<std::size_t>(gridDim.x)*Threads;
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*Threads+threadIdx.x;i<total;i+=stride) {
        const int n=i%s.state_dim,group=(i/s.state_dim)%s.groups,t=(i/s.state_dim/s.groups)%s.sequence,b=i/s.state_dim/s.groups/s.sequence;
        float q0=0,k0=0;
        if (status[b]==0) for (int h=group*(s.heads/s.groups);h<(group+1)*(s.heads/s.groups);++h) {
            const auto src=((static_cast<std::size_t>(b)*s.heads+h)*s.sequence+t)*s.state_dim+n;q0+=gq[src];k0+=gk[src];
        }
        dx.q[i]=q0;dx.k[i]=k0;
    }
    for (int i=blockIdx.x*Threads+threadIdx.x;i<s.heads*s.state_dim;i+=gridDim.x*Threads) {
        const int h=i/s.state_dim,n=i%s.state_dim;float q0=0,k0=0;
        for (int b=0;b<s.batch;++b) if (status[b]==0) {q0+=qb[(static_cast<std::size_t>(b)*s.heads+h)*s.state_dim+n];k0+=kb[(static_cast<std::size_t>(b)*s.heads+h)*s.state_dim+n];}
        dx.q_bias[i]=q0;dx.k_bias[i]=k0;
    }
    if (dx.d) for (int h=blockIdx.x*Threads+threadIdx.x;h<s.heads;h+=gridDim.x*Threads) {
        float value=0;for (int b=0;b<s.batch;++b) if (status[b]==0) value+=gd[static_cast<std::size_t>(b)*s.heads+h];dx.d[h]=value;
    }
}
template<bool Reduced>
__global__ __launch_bounds__(Threads) void validate_gradients(Shape s,Gradient dx,State di,const float* partials,int* status) {
    const int b=blockIdx.x,lane=threadIdx.x;
    if (status[b]!=0) return;
    __shared__ int flags[Threads];int bad=0;
    const auto bh=static_cast<std::size_t>(b)*s.heads;
    const auto th=bh*s.sequence;
    if constexpr (Reduced) {
        const auto begin=static_cast<std::size_t>(b)*s.sequence*s.groups*s.state_dim;
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*s.state_dim;i+=Threads) bad|=!isfinite(dx.q[begin+i])||!isfinite(dx.k[begin+i]);
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) bad|=!isfinite(dx.q_bias[i])||!isfinite(dx.k_bias[i]);
        if (dx.d) for (int h=lane;h<s.heads;h+=Threads) bad|=!isfinite(dx.d[h]);
    } else {
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.head_dim;i+=Threads) {
            bad|=!isfinite(dx.v[th*s.head_dim+i]);if (dx.z) bad|=!isfinite(dx.z[th*s.head_dim+i]);
        }
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads;i+=Threads) bad|=!isfinite(dx.adt[th+i])||!isfinite(dx.dt[th+i])||!isfinite(dx.trap[th+i]);
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.rotary_pairs;i+=Threads) bad|=!isfinite(dx.angles[th*s.rotary_pairs+i]);
        const auto expanded=static_cast<std::size_t>(s.batch)*s.heads*s.sequence*s.state_dim;
        const auto local=static_cast<std::size_t>(s.heads)*s.sequence*s.state_dim;
        for (std::size_t i=lane;i<local;i+=Threads) bad|=!isfinite(partials[bh*s.sequence*s.state_dim+i])||!isfinite(partials[expanded+bh*s.sequence*s.state_dim+i]);
        const auto bias=2*expanded+bh*s.state_dim;
        const auto bias_stride=static_cast<std::size_t>(s.batch)*s.heads*s.state_dim;
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) bad|=!isfinite(partials[bias+i])||!isfinite(partials[bias+bias_stride+i]);
        for (int h=lane;h<s.heads;h+=Threads) bad|=!isfinite(partials[2*expanded+2*bias_stride+bh+h]);
        for (int i=lane;i<s.heads*s.head_dim*s.state_dim;i+=Threads) bad|=!isfinite(di.ssm[bh*s.head_dim*s.state_dim+i]);
        for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) bad|=!isfinite(di.phase[bh*s.rotary_pairs+i]);
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) bad|=!isfinite(di.k[bh*s.state_dim+i]);
        for (int i=lane;i<s.heads*s.head_dim;i+=Threads) bad|=!isfinite(di.v[bh*s.head_dim+i]);
    }
    flags[lane]=bad;__syncthreads();
    if (lane==0) {for (int i=0;i<Threads;++i) bad|=flags[i];if (bad) status[b]=3;}
}
__global__ __launch_bounds__(Threads) void clear_failed_gradients(Shape s,Gradient dx,State di,const int* status) {
    const int b=blockIdx.x,lane=threadIdx.x;const auto bh=static_cast<std::size_t>(b)*s.heads,th=bh*s.sequence;
    if (status[b]!=0) {
        const auto begin=static_cast<std::size_t>(b)*s.sequence*s.groups*s.state_dim;
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.groups*s.state_dim;i+=Threads) {dx.q[begin+i]=0;dx.k[begin+i]=0;}
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.head_dim;i+=Threads) {dx.v[th*s.head_dim+i]=0;if (dx.z) dx.z[th*s.head_dim+i]=0;}
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads;i+=Threads) {dx.adt[th+i]=0;dx.dt[th+i]=0;dx.trap[th+i]=0;}
        for (std::size_t i=lane;i<static_cast<std::size_t>(s.sequence)*s.heads*s.rotary_pairs;i+=Threads) dx.angles[th*s.rotary_pairs+i]=0;
        for (int i=lane;i<s.heads*s.head_dim*s.state_dim;i+=Threads) di.ssm[bh*s.head_dim*s.state_dim+i]=0;
        for (int i=lane;i<s.heads*s.rotary_pairs;i+=Threads) di.phase[bh*s.rotary_pairs+i]=0;
        for (int i=lane;i<s.heads*s.state_dim;i+=Threads) di.k[bh*s.state_dim+i]=0;
        for (int i=lane;i<s.heads*s.head_dim;i+=Threads) di.v[bh*s.head_dim+i]=0;
    }
    // Shared parameter gradients are a group-level result: any rejected batch
    // invalidates publication for the entire op. Never hide a dropped sample.
    if (b==0) {
        bool bad=false;for (int batch=0;batch<s.batch;++batch) bad|=status[batch]!=0;
        if (bad) {
            for (int i=lane;i<s.heads*s.state_dim;i+=Threads) {dx.q_bias[i]=0;dx.k_bias[i]=0;}
            if (dx.d) for (int h=lane;h<s.heads;h+=Threads) dx.d[h]=0;
        }
    }
}
#endif
}

bool supported(const Shape& s) {
    if (!eligible(s)) return false;
#if defined(NSOS_GPU_BACKEND_HIP)
    int device=-1;cudaDeviceProp p{};
    if (cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&p,device)!=cudaSuccess) return false;
    const std::string arch=std::string(p.gcnArchName).substr(0,std::string(p.gcnArchName).find(':'));
    if (p.warpSize!=32||(arch!="gfx1100"&&arch!="gfx1101"&&arch!="gfx1102")||p.maxThreadsPerBlock<Threads||p.sharedMemPerBlock<45056||
        p.maxGridSize[0]<s.batch*s.heads) return false;
    bool compiled=false;for (const auto& d:gpu::enumerate_devices()) if (d.index==device) compiled=d.compiled;
    if (!compiled) return false;
    hipFuncAttributes a{},v{};
    return hipFuncGetAttributes(&a,reinterpret_cast<const void*>(scan))==hipSuccess&&
        hipFuncGetAttributes(&v,reinterpret_cast<const void*>(reverse_scan))==hipSuccess&&
        a.maxThreadsPerBlock>=Threads&&v.maxThreadsPerBlock>=Threads&&a.sharedSizeBytes<=p.sharedMemPerBlock&&v.sharedSizeBytes<=p.sharedMemPerBlock;
#else
    return false;
#endif
}
// Build host range lists dynamically so optional Z/D and zero-state inputs do
// not weaken overlap checks. They are metadata only, never device dereferences.
bool forward(const Shape& s,Input x,ConstState initial,State final,float* out,float* boundaries,int* status,const int* upstream) {
    if (!host_input(x)||(!empty(initial)&&!complete(initial))||!complete(as_const(final))||!supported(s)) return false;
    const auto bh=static_cast<std::size_t>(s.batch)*s.heads,th=bh*s.sequence;
    const auto qn=static_cast<std::size_t>(s.batch)*s.sequence*s.groups*s.state_dim*sizeof(float),vn=th*s.head_dim*sizeof(float),an=th*s.rotary_pairs*sizeof(float);
    const auto hn=th*sizeof(float),bn=static_cast<std::size_t>(s.heads)*s.state_dim*sizeof(float),sn=bh*s.head_dim*s.state_dim*sizeof(float);
    if (!disjoint({{x.q,qn},{x.k,qn},{x.v,vn},{x.adt,hn},{x.dt,hn},{x.trap,hn},{x.angles,an},{x.q_bias,bn},{x.k_bias,bn},{x.valid,static_cast<std::size_t>(s.batch)*sizeof(int)}},
        {{out,vn},{boundaries,boundary_elements(s)*sizeof(float)},{status,static_cast<std::size_t>(s.batch)*sizeof(int)},
         {final.phase,bh*s.rotary_pairs*sizeof(float)},{final.ssm,sn},{final.k,bh*s.state_dim*sizeof(float)},{final.v,bh*s.head_dim*sizeof(float)}})) return false;
    auto separated_optional=[&](const void* ptr,std::size_t count) {
        return !ptr||disjoint({{ptr,count}},{{out,vn},{boundaries,boundary_elements(s)*sizeof(float)},{status,static_cast<std::size_t>(s.batch)*sizeof(int)},
            {final.phase,bh*s.rotary_pairs*sizeof(float)},{final.ssm,sn},{final.k,bh*s.state_dim*sizeof(float)},{final.v,bh*s.head_dim*sizeof(float)}});
    };
    if (!separated_optional(upstream,static_cast<std::size_t>(s.batch)*sizeof(int))||
        !separated_optional(x.z,vn)||!separated_optional(x.d,s.heads*sizeof(float))||
        !separated_optional(initial.phase,bh*s.rotary_pairs*sizeof(float))||!separated_optional(initial.ssm,sn)||
        !separated_optional(initial.k,bh*s.state_dim*sizeof(float))||!separated_optional(initial.v,bh*s.head_dim*sizeof(float))) return false;
#if defined(NSOS_GPU_BACKEND_HIP)
    const auto stream=gpu::current_stream();preflight<false><<<s.batch,Threads,0,stream>>>(s,x,initial,nullptr,{},status,upstream);
    if (cudaGetLastError()!=cudaSuccess) return false;
    scan<<<s.batch*s.heads,Threads,0,stream>>>(s,x,initial,final,out,boundaries,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    finalize_forward<<<s.batch,Threads,0,stream>>>(s,final,out,status);return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}
bool backward(const Shape& s,Input x,ConstState initial,const float* dy,ConstState seed,Gradient dx,
    State di,const float* boundaries,float* replay,float* partials,int* status) {
    if (!host_input(x)||(!empty(initial)&&!complete(initial))||(!empty(seed)&&!complete(seed))||!complete(as_const(di))||
        !dx.q||!dx.k||!dx.v||!dx.adt||!dx.dt||!dx.trap||!dx.angles||!dx.q_bias||!dx.k_bias||
        ((x.z==nullptr)!=(dx.z==nullptr))||((x.d==nullptr)!=(dx.d==nullptr))||!supported(s)) return false;
    // Raw ABI ranges are checked before any enqueue. Owning model/tape
    // integration must additionally guarantee capacities and immutable inputs.
    const auto bh=static_cast<std::size_t>(s.batch)*s.heads,th=bh*s.sequence;
    const auto qn=static_cast<std::size_t>(s.batch)*s.sequence*s.groups*s.state_dim*sizeof(float),vn=th*s.head_dim*sizeof(float),an=th*s.rotary_pairs*sizeof(float),hn=th*sizeof(float);
    const auto bn=static_cast<std::size_t>(s.heads)*s.state_dim*sizeof(float),sn=bh*s.head_dim*s.state_dim*sizeof(float);
    // Fixed arrays avoid allocation; optional ranges get omitted rather than
    // pretending null pointers authorize aliasing or missing optional adjoints.
    Range reads[24],writes[19];int nr=0,nw=0;
    auto r=[&](const void* p,std::size_t n) {if (p) reads[nr++]={p,n};};
    auto w=[&](void* p,std::size_t n) {if (p) writes[nw++]={p,n};};
    r(x.q,qn);r(x.k,qn);r(x.v,vn);r(x.z,vn);r(x.adt,hn);r(x.dt,hn);r(x.trap,hn);r(x.angles,an);r(x.q_bias,bn);r(x.k_bias,bn);r(x.d,s.heads*sizeof(float));r(x.valid,s.batch*sizeof(int));r(dy,vn);r(boundaries,boundary_elements(s)*sizeof(float));
    r(initial.phase,bh*s.rotary_pairs*sizeof(float));r(initial.ssm,sn);r(initial.k,bh*s.state_dim*sizeof(float));r(initial.v,bh*s.head_dim*sizeof(float));
    r(seed.phase,bh*s.rotary_pairs*sizeof(float));r(seed.ssm,sn);r(seed.k,bh*s.state_dim*sizeof(float));r(seed.v,bh*s.head_dim*sizeof(float));
    w(dx.q,qn);w(dx.k,qn);w(dx.v,vn);w(dx.z,vn);w(dx.adt,hn);w(dx.dt,hn);w(dx.trap,hn);w(dx.angles,an);w(dx.q_bias,bn);w(dx.k_bias,bn);w(dx.d,s.heads*sizeof(float));
    w(di.phase,bh*s.rotary_pairs*sizeof(float));w(di.ssm,sn);w(di.k,bh*s.state_dim*sizeof(float));w(di.v,bh*s.head_dim*sizeof(float));
    w(replay,replay_elements(s)*sizeof(float));w(partials,gradient_partial_elements(s)*sizeof(float));w(status,s.batch*sizeof(int));
    if (!dy||!boundaries||!replay||!partials||!status) return false;
    // Also reject overlapping writes with one another, including optional Z/D.
    auto good=[](Range a) {const auto p=reinterpret_cast<std::uintptr_t>(a.data);return p&&p%4==0&&a.bytes&&p<=std::numeric_limits<std::uintptr_t>::max()-a.bytes;};
    auto overlaps=[](Range a,Range b) {const auto p=reinterpret_cast<std::uintptr_t>(a.data),q=reinterpret_cast<std::uintptr_t>(b.data);return p<q+b.bytes&&q<p+a.bytes;};
    for (int i=0;i<nr;++i) if (!good(reads[i])) return false;
    for (int i=0;i<nw;++i) {if (!good(writes[i])) return false;for (int j=0;j<nr;++j) if (overlaps(writes[i],reads[j])) return false;for (int j=i+1;j<nw;++j) if (overlaps(writes[i],writes[j])) return false;}
#if defined(NSOS_GPU_BACKEND_HIP)
    const auto stream=gpu::current_stream();preflight<true><<<s.batch,Threads,0,stream>>>(s,x,initial,dy,seed,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    reverse_scan<<<s.batch*s.heads,Threads,0,stream>>>(s,x,initial,dy,seed,dx,di,boundaries,replay,partials,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    validate_gradients<false><<<s.batch,Threads,0,stream>>>(s,dx,di,partials,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    const auto n=static_cast<std::size_t>(s.batch)*s.sequence*s.groups*s.state_dim;
    const int blocks=static_cast<int>((n-1)/Threads+1>4096?4096:(n-1)/Threads+1);
    reduce_gradients<<<blocks,Threads,0,stream>>>(s,dx,partials,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    validate_gradients<true><<<s.batch,Threads,0,stream>>>(s,dx,di,partials,status);
    if (cudaGetLastError()!=cudaSuccess) return false;
    clear_failed_gradients<<<s.batch,Threads,0,stream>>>(s,dx,di,status);return cudaGetLastError()==cudaSuccess;
#else
    return false;
#endif
}
}
