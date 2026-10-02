#pragma once
// Exact affine decomposition of the Mamba-3 trapezoidal recurrence. This is
// intentionally independent of the Mamba-2 SSD kernel and its state layout.
namespace nsos::mamba3_block::parallel_detail {
constexpr int tile=32, lanes=128;
__device__ inline void fail(float* status,int b,int code=3) { atomicExch(status+b,float(code)); }
__device__ inline const float* row(Shape s,const float* p,int b,int t) {
    return p+(std::size_t(b)*s.sequence+t)*s.width();
}
struct Coeff {float alpha,beta,gamma,dt,a,lam,act;};
__device__ inline Coeff coeff(Shape s,Layout l,const float* raw,const float* w,int h) {
    const int off=2*s.inner()+2*s.bc();
    Coeff c;c.act=detail::heavy(raw[off+s.heads+h]);c.a=-fmaxf(c.act,float(s.a_floor));
    c.dt=detail::softplus(raw[off+h]+w[l.dt+h]);c.alpha=expf(c.a*c.dt);
    c.lam=detail::sigmoid(raw[off+2*s.heads+h]);
    c.beta=(1-c.lam)*c.dt*c.alpha;c.gamma=c.lam*c.dt;return c;
}
// One immutable FP32 SoA coefficient set per BH/token. The evaluator above
// is invoked only by this preparation kernel, never in PN/Flash replay loops.
__device__ inline Coeff cached_coeff(Shape s,const float* cache,int bh,int t) {
    const auto count=std::size_t(s.batch)*s.heads*s.sequence,i=std::size_t(bh)*s.sequence+t;
    return {cache[i],cache[count+i],cache[2*count+i],cache[3*count+i],cache[4*count+i],cache[5*count+i],cache[6*count+i]};
}
__global__ void coefficient_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,float* cache,float* status) {
    const auto count=std::size_t(s.batch)*s.heads*s.sequence,i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const int t=int(i%s.sequence),bh=int(i/s.sequence),b=bh/s.heads,h=bh%s.heads,len=valid[b];
    if(len<0||len>s.sequence) {fail(status,b,1);return;}
    if(t>=len) return; // Do not evaluate or read poison padding.
    const auto c=coeff(s,l,row(s,p,b,t),w,h);
    cache[i]=c.alpha;cache[count+i]=c.beta;cache[2*count+i]=c.gamma;cache[3*count+i]=c.dt;
    cache[4*count+i]=c.a;cache[5*count+i]=c.lam;cache[6*count+i]=c.act;
    if(!isfinite(c.alpha)||!isfinite(c.beta)||!isfinite(c.gamma)||!isfinite(c.dt)||!isfinite(c.a)||!isfinite(c.lam)||!isfinite(c.act)) fail(status,b);
}
__device__ inline float reduce(float v,float* shared) {
    const int n=threadIdx.x;shared[n]=v;__syncthreads();
    for(int stride=lanes/2;stride;stride/=2) {if(n<stride) shared[n]+=shared[n+stride];__syncthreads();}
    v=shared[0];__syncthreads();return v;
}
__device__ inline double reduce_double(double v,double* shared) {
    const int n=threadIdx.x;shared[n]=v;__syncthreads();
    for(int stride=lanes/2;stride;stride/=2) {if(n<stride) shared[n]+=shared[n+stride];__syncthreads();}
    v=shared[0];__syncthreads();return v;
}
// Every lane reads the old level before any lane publishes the new level.
// compose(right,left) = (right.a*left.a, right.b+right.a*left.b).
__device__ inline void affine_scan(float& a,float& v,float* aa,float* vv,int lane,int base) {
    aa[threadIdx.x]=a;vv[threadIdx.x]=v;__syncthreads();
    for(int offset=1;offset<tile;offset*=2) {
        float na=a,nv=v;if(lane>=offset) {const float la=aa[base+lane-offset],lv=vv[base+lane-offset];na=a*la;nv=v+a*lv;}
        __syncthreads();a=na;v=nv;aa[threadIdx.x]=a;vv[threadIdx.x]=v;__syncthreads();
    }
}
__device__ inline float wrap(float x) {return x-float(detail::tau)*floorf(x/float(detail::tau));}
__global__ void phase_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,State<float> final,Trace<float> tr,float* status) {
    // Phase storage is part of the public streaming/BPTT state. A circular
    // parallel prefix is mathematically equivalent, but can cross the seam
    // after reassociation of sub-ULP updates that the FP32 reference retains.
    // Preserve its token order, parallelizing independent phase pairs/heads.
    const int bh=blockIdx.x,h=bh%s.heads,b=bh/s.heads,a=threadIdx.x;
    if(a>=s.rotary_pairs) return;
    const int len=valid[b];if(len<0||len>s.sequence) {fail(status,b,1);return;}
    float phase=detail::value(initial.phase,std::size_t(bh)*s.rotary_pairs+a);
    if(!isfinite(phase)) {fail(status,b,2);return;}
    for(int t=0;t<len;++t) {
        const float* raw=row(s,p,b,t);
        phase=wrap(phase+float(detail::pi)*tanhf(raw[s.width()-s.rotary_pairs+a])*cached_coeff(s,tr.coefficients,bh,t).dt);
        tr.phase[(std::size_t(bh)*s.sequence+t)*s.rotary_pairs+a]=phase;
        if(!isfinite(phase)) fail(status,b);
    }
    final.phase[std::size_t(bh)*s.rotary_pairs+a]=phase;
}
__global__ void rotate_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,State<float> final,Trace<float> tr,float* status) {
    const int t=blockIdx.x%s.sequence,bh=blockIdx.x/s.sequence,h=bh%s.heads,b=bh/s.heads,n=threadIdx.x;
    if(t>=valid[b]) return;
    __shared__ double red[lanes];__shared__ float vals[lanes];
    const float* raw=row(s,p,b,t);const auto off=(std::size_t(bh)*s.sequence+t)*s.rank*s.state_dim;
    for(int r=0;r<s.rank;++r) for(int kind=0;kind<2;++kind) {
        const float* bc=raw+2*s.inner()+(kind?0:s.bc())+(r*s.groups+h/(s.heads/s.groups))*s.state_dim;
        const double x=n<s.state_dim?bc[n]:0;
        const double sq=reduce_double(x*x,red);const float inv=float(1/sqrt(sq/s.state_dim+s.norm_eps));
        vals[n]=n<s.state_dim?bc[n]*inv*w[(kind?l.bnorm:l.cnorm)+n]+w[(kind?l.bbias:l.cbias)+(h*s.rank+r)*s.state_dim+n]:0;
        __syncthreads();float v=vals[n];
        for(int a=0;a<s.rotary_pairs;++a) {const int i=detail::coord(s,a,false),j=detail::coord(s,a,true);
            if(n==i||n==j) {const float ph=tr.phase[(std::size_t(bh)*s.sequence+t)*s.rotary_pairs+a],c=cosf(ph),sn=sinf(ph);
                v=n==i?vals[i]*c-vals[j]*sn:vals[i]*sn+vals[j]*c;}}
        if(n<s.state_dim) {(kind?tr.k:tr.q)[off+r*s.state_dim+n]=v;if(!isfinite(v)) fail(status,b);
            if(kind&&t==valid[b]-1) final.k[std::size_t(bh)*s.rank*s.state_dim+r*s.state_dim+n]=v;}
        __syncthreads();
    }
    if(t==valid[b]-1&&n<s.head_dim) final.v[std::size_t(bh)*s.head_dim+n]=raw[s.inner()+h*s.head_dim+n];
}
__device__ inline float drive(Shape s,Layout l,const float* p,const float* w,Trace<const float> tr,
        State<const float> initial,int b,int h,int t,int pp,int n,Coeff c) {
    const auto bh=std::size_t(b)*s.heads+h;const float* raw=row(s,p,b,t);
    const float pv=t?row(s,p,b,t-1)[s.inner()+h*s.head_dim+pp]:detail::value(initial.v,bh*s.head_dim+pp);
    float prev=0,curr=0;
    for(int r=0;r<s.rank;++r) {const float px=s.mimo?w[l.x+(h*s.rank+r)*s.head_dim+pp]:1;
        const float pk=t?tr.k[((bh*s.sequence+t-1)*s.rank+r)*s.state_dim+n]:detail::value(initial.k,(bh*s.rank+r)*s.state_dim+n);
        prev+=pk*pv*px;curr+=tr.k[((bh*s.sequence+t)*s.rank+r)*s.state_dim+n]*raw[s.inner()+h*s.head_dim+pp]*px;}
    return c.beta*prev+c.gamma*curr;
}
__host__ __device__ inline Trace<const float> constant(Trace<float> x) {return {x.history,x.q,x.k,x.phase,x.readout,x.parallel,x.checkpoints,x.coefficients,x.replay_lds,x.hierarchical,x.hierarchy_a,x.hierarchy_b};}
__device__ inline std::size_t history_at(Shape s,Trace<const float> tr,std::size_t bh,int t,int cell) {
    const int slots=tr.checkpoints?(s.sequence+tile-1)/tile+1:s.sequence+1;
    return (bh*slots+t)*s.head_dim*s.state_dim+cell;
}
__global__ void state_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,State<float> final,Trace<float> trace,float* status) {
    const std::size_t cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const auto id=std::size_t(blockIdx.x)*4+threadIdx.x/tile;const int lane=threadIdx.x%tile,base=threadIdx.x-lane;
    const int cell=int(id%(s.head_dim*s.state_dim)),n=cell%s.state_dim,pp=cell/s.state_dim;
    const auto bh=id/(s.head_dim*s.state_dim);const int h=int(bh%s.heads),b=int(bh/s.heads);
    __shared__ float aa[lanes],vv[lanes];auto tr=constant(trace);
    float carry=id<cells?detail::value(initial.ssm,id):0;
    // Blocks include a padded final group: all groups participate in barriers.
    const int len=id<cells?valid[b]:0;
    if(!lane&&id<cells) {trace.history[history_at(s,tr,bh,0,cell)]=carry;if(!isfinite(carry)) fail(status,b,2);}
    const int maximum=s.sequence;for(int start=0;start<maximum;start+=tile) {
        const int t=start+lane;float alpha=1,v=0;
        if(id<cells&&t<len) {auto c=cached_coeff(s,tr.coefficients,int(bh),t);alpha=c.alpha;v=drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);}
        affine_scan(alpha,v,aa,vv,lane,base);const float value=alpha*carry+v;
        if(id<cells&&t<len&&!tr.checkpoints) {trace.history[history_at(s,tr,bh,t+1,cell)]=value;if(!isfinite(value)) fail(status,b);}
        __syncthreads();const int end=std::max(0,std::min(tile,len-start));
        if(end) carry=aa[base+end-1]*carry+vv[base+end-1];
        if(!lane&&id<cells&&tr.checkpoints) {trace.history[history_at(s,tr,bh,start/tile+1,cell)]=carry;if(!isfinite(carry)) fail(status,b);}
        __syncthreads();
    }
    if(!lane&&id<cells) final.ssm[id]=carry;
}
__device__ inline float state_value(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,Trace<const float> tr,int b,int h,int t,int pp,int n) {
    const auto bh=std::size_t(b)*s.heads+h;const int cell=pp*s.state_dim+n;
    if(!tr.checkpoints) return tr.history[history_at(s,tr,bh,t+1,cell)];
    const int start=(t/tile)*tile;float value=tr.history[history_at(s,tr,bh,start/tile,cell)];
    for(int j=start;j<=t;++j) {const auto c=cached_coeff(s,tr.coefficients,int(bh),j);value=c.alpha*value+drive(s,l,p,w,tr,initial,b,h,j,pp,n,c);}
    return value;
}
__global__ void readout_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,Trace<float> trace,float* status) {
    const auto i=std::size_t(blockIdx.x)*lanes+threadIdx.x;
    const std::size_t total=std::size_t(s.batch)*s.heads*s.sequence*s.rank*s.head_dim;if(i>=total) return;
    const int pp=int(i%s.head_dim),r=int(i/s.head_dim%s.rank),t=int(i/(s.head_dim*s.rank)%s.sequence);
    const auto bh=i/(std::size_t(s.head_dim)*s.rank*s.sequence);const int h=int(bh%s.heads),b=int(bh/s.heads);
    if(t>=valid[b]) return;auto tr=constant(trace);float acc=0;
    for(int n=0;n<s.state_dim;++n) acc+=state_value(s,l,p,w,valid,initial,tr,b,h,t,pp,n)*tr.q[((bh*s.sequence+t)*s.rank+r)*s.state_dim+n];
    const float px=s.mimo?w[l.x+(h*s.rank+r)*s.head_dim+pp]:1;
    trace.readout[i]=acc+w[l.d+h]*row(s,p,b,t)[s.inner()+h*s.head_dim+pp]*px;
    if(!isfinite(trace.readout[i])) fail(status,b);
}
// Four consecutive tokens and four P rows share one bounded state replay.
// The 129-float N stride avoids four-way LDS bank conflicts across P rows.
// Dot products keep their sequential-N FP32 order; no rank sees a different
// replay or sum tree. Only the Flash provider uses this path.
constexpr int flash_readout_t=4,flash_readout_p=4,flash_readout_n_stride=129;
__global__ void flash_readout_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,Trace<float> trace,float* status) {
    const int ptiles=(s.head_dim+flash_readout_p-1)/flash_readout_p,ttiles=(s.sequence+flash_readout_t-1)/flash_readout_t;
    const int ptile=blockIdx.x%ptiles,ttile=(blockIdx.x/ptiles)%ttiles,bh=blockIdx.x/(std::size_t(ptiles)*ttiles);
    const int b=bh/s.heads,h=bh%s.heads,n=threadIdx.x,tbase=ttile*flash_readout_t,pbase=ptile*flash_readout_p,len=valid[b];
    if(len<0||len>s.sequence||tbase>=len) return;
    __shared__ float state_tile[flash_readout_t][flash_readout_p][flash_readout_n_stride];
    const auto tr=constant(trace);const int begin=(tbase/tile)*tile;
    for(int localp=0;localp<flash_readout_p;++localp) {
        const int pp=pbase+localp;
        if(pp<s.head_dim&&n<s.state_dim) {
            float value=tr.history[history_at(s,tr,bh,begin/tile,pp*s.state_dim+n)];
            for(int t=begin;t<tbase;++t) {const auto c=cached_coeff(s,tr.coefficients,bh,t);value=c.alpha*value+drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);}
            for(int localt=0;localt<flash_readout_t;++localt) {const int t=tbase+localt;
                if(t<len) {const auto c=cached_coeff(s,tr.coefficients,bh,t);value=c.alpha*value+drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);
                    state_tile[localt][localp][n]=value;}}
        }
    }
    __syncthreads();
    const int owner=threadIdx.x;
    if(owner<flash_readout_t*flash_readout_p*s.rank) {
        const int localp=owner%flash_readout_p,r=(owner/flash_readout_p)%s.rank,localt=owner/(flash_readout_p*s.rank);
        const int pp=pbase+localp,t=tbase+localt;
        if(pp<s.head_dim&&t<len) {
            const auto ti=std::size_t(bh)*s.sequence+t;float acc=0;
            for(int j=0;j<s.state_dim;++j) acc+=state_tile[localt][localp][j]*tr.q[(ti*s.rank+r)*s.state_dim+j];
            const float px=s.mimo?w[l.x+(h*s.rank+r)*s.head_dim+pp]:1;
            const float value=acc+w[l.d+h]*row(s,p,b,t)[s.inner()+h*s.head_dim+pp]*px;
            trace.readout[(ti*s.rank+r)*s.head_dim+pp]=value;if(!isfinite(value)) fail(status,b);
        }
    }
}
__global__ void output_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        Trace<const float> tr,float* y,float* status) {
    const int t=blockIdx.x%s.sequence,bh=blockIdx.x/s.sequence,h=bh%s.heads,b=bh/s.heads,pp=threadIdx.x;
    if(t>=valid[b]) return;__shared__ double red[lanes];float output=0;const float* raw=row(s,p,b,t);
    for(int r=0;r<s.rank;++r) {const float v=pp<s.head_dim?tr.readout[((std::size_t(bh)*s.sequence+t)*s.rank+r)*s.head_dim+pp]:0;
        const double sq=reduce_double(double(v)*v,red);const float inv=s.out_norm?float(1/sqrt(sq/s.head_dim+s.norm_eps)):1;
        if(pp<s.head_dim) {const int hp=h*s.head_dim+pp,rp=(h*s.rank+r)*s.head_dim+pp;const float z=raw[hp]*(s.mimo?w[l.z+rp]:1);
            output+=v*inv*(s.out_norm?w[l.norm+hp]:1)*z*detail::sigmoid(z)*(s.mimo?w[l.o+rp]:1);}}
    if(pp<s.head_dim) {y[(std::size_t(b)*s.sequence+t)*s.inner()+h*s.head_dim+pp]=output;if(!isfinite(output)) fail(status,b);}
}
__global__ void empty_state_kernel(Shape s,const int* valid,State<const float> initial,State<float> final,float* status) {
    const int bh=blockIdx.x,h=bh%s.heads,b=bh/s.heads,n=threadIdx.x;
    for(int r=0;r<s.rank;++r) if(n<s.state_dim) {const float v=detail::value(initial.k,(std::size_t(bh)*s.rank+r)*s.state_dim+n);
        if(!isfinite(v)) fail(status,b,2);if(!valid[b]) final.k[(std::size_t(bh)*s.rank+r)*s.state_dim+n]=v;}
    if(n<s.head_dim) {const float v=detail::value(initial.v,std::size_t(bh)*s.head_dim+n);if(!isfinite(v)) fail(status,b,2);if(!valid[b]) final.v[std::size_t(bh)*s.head_dim+n]=v;}
}
__device__ inline Layout local_layout(Shape s) {s.heads=1;s.groups=1;return Layout(s);}
__device__ inline std::size_t token(Shape s,int bh,int t) {return std::size_t(bh)*s.sequence+t;}
__global__ void output_backward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        Trace<const float> tr,const float* dy,BackwardWorkspace ws,float* status) {
    const int t=blockIdx.x%s.sequence,bh=blockIdx.x/s.sequence,h=bh%s.heads,b=bh/s.heads,pp=threadIdx.x;
    if(t>=valid[b]) return;const auto ll=local_layout(s);const auto ti=token(s,bh,t);float* tp=ws.token_parameters+ti*ll.total;
    const float* raw=row(s,p,b,t);__shared__ double red[lanes];
    for(int r=0;r<s.rank;++r) {const auto rp=(h*s.rank+r)*s.head_dim+pp;
        const float yy=pp<s.head_dim?tr.readout[(ti*s.rank+r)*s.head_dim+pp]:0;
        const double sq=reduce_double(double(yy)*yy,red);const float inv=s.out_norm?float(1/sqrt(sq/s.head_dim+s.norm_eps)):1;
        float gy=0;
        if(pp<s.head_dim) {const float upstream=dy[(std::size_t(b)*s.sequence+t)*s.inner()+h*s.head_dim+pp];
            if(!isfinite(upstream)) fail(status,b,2);const float z=raw[h*s.head_dim+pp]*(s.mimo?w[l.z+rp]:1),sg=detail::sigmoid(z),gate=z*sg;
            const float norm=s.out_norm?w[l.norm+h*s.head_dim+pp]:1,po=s.mimo?w[l.o+rp]:1;
            gy=upstream*po*gate*norm;
            if(s.mimo) {tp[ll.z+r*s.head_dim+pp]=upstream*po*yy*inv*norm*sg*(1+z*(1-sg))*raw[h*s.head_dim+pp];
                tp[ll.o+r*s.head_dim+pp]=upstream*yy*inv*norm*gate;}
            if(s.out_norm) tp[ll.norm+pp]+=upstream*po*gate*yy*inv;
            ws.dz[ti*s.head_dim+pp]+=upstream*po*yy*inv*norm*sg*(1+z*(1-sg))*(s.mimo?w[l.z+rp]:1);}
        const double cross=reduce_double(double(gy)*yy,red);
        if(pp<s.head_dim) ws.gy[(ti*s.rank+r)*s.head_dim+pp]=s.out_norm?float(double(inv)*(double(gy)-double(yy)*inv*inv*cross/s.head_dim)):gy;
    }
}
__device__ inline float injection(Shape s,Trace<const float> tr,BackwardWorkspace ws,int bh,int t,int pp,int n) {
    float value=0;const auto ti=token(s,bh,t);
    for(int r=0;r<s.rank;++r) value+=ws.gy[(ti*s.rank+r)*s.head_dim+pp]*tr.q[(ti*s.rank+r)*s.state_dim+n];return value;
}
__global__ void reverse_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,Trace<const float> tr,
        State<const float> seed,State<float> di,BackwardWorkspace ws,float* status) {
    const std::size_t cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const auto id=std::size_t(blockIdx.x)*4+threadIdx.x/tile;const int lane=threadIdx.x%tile,base=threadIdx.x-lane;
    const int cell=int(id%(s.head_dim*s.state_dim)),n=cell%s.state_dim,pp=cell/s.state_dim;
    const auto bh=id/(s.head_dim*s.state_dim);const int h=int(bh%s.heads),b=int(bh/s.heads);
    __shared__ float aa[lanes],vv[lanes];float carry=id<cells?detail::value(seed.ssm,id):0;
    const int len=id<cells?valid[b]:0,chunks=(s.sequence+tile-1)/tile;
    if(!lane&&id<cells) {ws.reverse[history_at(s,tr,bh,tr.checkpoints?chunks:len,cell)]=carry;if(!isfinite(carry)) fail(status,b,2);}
    for(int chunk=chunks-1;chunk>=0;--chunk) {
        const int begin=chunk*tile,end=std::min(begin+tile,len),t=end-1-lane;float a=1,v=0;
        if(id<cells&&t>=begin&&t<len) {a=cached_coeff(s,tr.coefficients,int(bh),t).alpha;v=a*injection(s,tr,ws,int(bh),t,pp,n);}
        affine_scan(a,v,aa,vv,lane,base);const float dh=a*carry+v;
        if(id<cells&&t>=begin&&t<len&&!tr.checkpoints) ws.reverse[history_at(s,tr,bh,t,cell)]=dh;
        __syncthreads();const int count=std::max(0,end-begin);if(count) carry=aa[base+count-1]*carry+vv[base+count-1];
        if(!lane&&id<cells&&tr.checkpoints) ws.reverse[history_at(s,tr,bh,chunk,cell)]=carry;__syncthreads();
    }
    if(!lane&&id<cells) {di.ssm[id]=carry;if(!isfinite(carry)) fail(status,b);}
}
__device__ inline float gh_value(Shape s,Layout l,const float* p,const float* w,const int* valid,
        Trace<const float> tr,State<const float> seed,BackwardWorkspace ws,int b,int h,int t,int pp,int n) {
    const int bh=b*s.heads+h,cell=pp*s.state_dim+n,len=valid[b];
    if(t>=len) return detail::value(seed.ssm,std::size_t(bh)*s.head_dim*s.state_dim+cell);
    float dh;
    if(!tr.checkpoints) dh=ws.reverse[history_at(s,tr,bh,t+1,cell)];
    else {const int end=std::min((t/tile+1)*tile,len);dh=ws.reverse[history_at(s,tr,bh,t/tile+1,cell)];
        for(int j=end-1;j>t;--j) dh=cached_coeff(s,tr.coefficients,bh,j).alpha*(dh+injection(s,tr,ws,bh,j,pp,n));}
    return dh+injection(s,tr,ws,bh,t,pp,n);
}
// Each token/head owns its parameter contributions. N lanes cooperate on all
// radial/SSM reductions; shared groups are gathered by a separate fixed-order
// kernel. No floating atomic additions occur anywhere on this training path.
__global__ void token_backward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,Trace<const float> tr,State<const float> seed,State<float> di,
        BackwardWorkspace ws,float* status) {
    const int t=blockIdx.x%s.sequence,bh=blockIdx.x/s.sequence,h=bh%s.heads,b=bh/s.heads,n=threadIdx.x;
    const int N=s.state_dim,P=s.head_dim,R=s.rank;if(t>=valid[b]) return;
    __shared__ float red[lanes],dqr[8*lanes],dkr[8*lanes];__shared__ double dred[lanes];
    const auto ti=token(s,bh,t);const auto ll=local_layout(s);float* tp=ws.token_parameters+ti*ll.total;
    const float* raw=row(s,p,b,t);const auto c=cached_coeff(s,tr.coefficients,bh,t);
    float dq[8]{},dk[8]{},da=0,db=0,dc=0;
    for(int pp=0;pp<P;++pp) {
        const float vx=raw[s.inner()+h*P+pp],pv=t?row(s,p,b,t-1)[s.inner()+h*P+pp]:detail::value(initial.v,std::size_t(bh)*P+pp);
        const float gh=n<N?gh_value(s,l,p,w,valid,tr,seed,ws,b,h,t,pp,n):0;
        const float state=n<N?state_value(s,l,p,w,valid,initial,tr,b,h,t,pp,n):0;
        const float prevstate=n<N?(t?state_value(s,l,p,w,valid,initial,tr,b,h,t-1,pp,n):detail::value(initial.ssm,std::size_t(bh)*P*N+pp*N+n)):0;
        da+=gh*prevstate;float dx=0,initial_v=0;
        const float nextgh=t+1<valid[b]&&n<N?gh_value(s,l,p,w,valid,tr,seed,ws,b,h,t+1,pp,n):0;
        const auto nextc=t+1<valid[b]?cached_coeff(s,tr.coefficients,bh,t+1):c;
        for(int r=0;r<R;++r) {const int rp=(h*R+r)*P+pp;const float px=s.mimo?w[l.x+rp]:1;
            const float kt=n<N?tr.k[(ti*R+r)*N+n]:0,pk=n<N?(t?tr.k[((ti-1)*R+r)*N+n]:detail::value(initial.k,(std::size_t(bh)*R+r)*N+n)):0;
            const float adj=ws.gy[(ti*R+r)*P+pp];dq[r]+=adj*state;
            db+=gh*pk*pv*px;dc+=gh*kt*vx*px;dk[r]+=gh*c.gamma*vx*px;
            if(t+1<valid[b]) dk[r]+=nextgh*nextc.beta*vx*px;
            dx+=gh*c.gamma*kt*px+nextgh*(t+1<valid[b]?nextc.beta:0)*kt*px;
            const float gx=reduce(n<N?gh*(c.gamma*kt*vx+c.beta*pk*pv):0,red);
            if(!n&&s.mimo) tp[ll.x+r*P+pp]=gx+adj*w[l.d+h]*vx;
            initial_v+=gh*c.beta*pk*px;
            if(!n) {tp[ll.d]+=adj*vx*px;ws.dx[ti*P+pp]+=adj*w[l.d+h]*px;}
            if(!t&&n<N) di.k[(std::size_t(bh)*R+r)*N+n]+=gh*c.beta*pv*px;
        }
        const float dvalue=reduce(n<N?dx:0,red),div=reduce(n<N?initial_v:0,red);
        if(!n) {ws.dx[ti*P+pp]+=dvalue+(t==valid[b]-1?detail::value(seed.v,std::size_t(bh)*P+pp):0);
            if(!t) di.v[std::size_t(bh)*P+pp]=div;}
    }
    if(t==valid[b]-1&&n<N) for(int r=0;r<R;++r) dk[r]+=detail::value(seed.k,(std::size_t(bh)*R+r)*N+n);
    const float dda=reduce(n<N?da:0,red),ddb=reduce(n<N?db:0,red),ddc=reduce(n<N?dc:0,red);
    if(!n) {const float dadt=(dda+ddb*(1-c.lam)*c.dt)*c.alpha;
        ws.ddt[ti]=ddb*(1-c.lam)*c.alpha+ddc*c.lam+c.a*dadt;
        ws.da[ti]=c.act>float(s.a_floor)?-dadt*c.dt*(raw[2*s.inner()+2*s.bc()+s.heads+h]>=0?1:c.act*c.act):0;
        ws.dtrap[ti]=(-ddb*c.dt*c.alpha+ddc*c.dt)*c.lam*(1-c.lam);}
    for(int r=0;r<R;++r) {dqr[r*lanes+n]=dq[r];dkr[r*lanes+n]=dk[r];}__syncthreads();
    for(int a=0;a<s.rotary_pairs;++a) {const int i=detail::coord(s,a,false),j=detail::coord(s,a,true);float gp=0;
        if(n==i) for(int r=0;r<R;++r) {const float* q=tr.q+(ti*R+r)*N,*k=tr.k+(ti*R+r)*N;
            gp+=-dqr[r*lanes+i]*q[j]+dqr[r*lanes+j]*q[i]-dkr[r*lanes+i]*k[j]+dkr[r*lanes+j]*k[i];}
        if(n==i) ws.phase[ti*s.rotary_pairs+a]=gp;
    }
    for(int r=0;r<R;++r) {
        float aq=dq[r],ak=dk[r];
        for(int a=0;a<s.rotary_pairs;++a) {const int i=detail::coord(s,a,false),j=detail::coord(s,a,true);
            if(n==i||n==j) {const float phase=tr.phase[ti*s.rotary_pairs+a],co=cosf(phase),sn=sinf(phase);
                aq=n==i?dqr[r*lanes+i]*co+dqr[r*lanes+j]*sn:-dqr[r*lanes+i]*sn+dqr[r*lanes+j]*co;
                ak=n==i?dkr[r*lanes+i]*co+dkr[r*lanes+j]*sn:-dkr[r*lanes+i]*sn+dkr[r*lanes+j]*co;}}
        for(int kind=0;kind<2;++kind) {const float* bc=raw+2*s.inner()+(kind?0:s.bc())+(r*s.groups+h/(s.heads/s.groups))*N;
            const float adj=kind?ak:aq;const auto norm=kind?l.bnorm:l.cnorm,bias=(kind?ll.bbias:ll.cbias)+r*N;
            const double sq=reduce_double(n<N?double(bc[n])*bc[n]:0,dred);
            const double cross=reduce_double(n<N?double(bc[n])*w[norm+n]*adj:0,dred),inv=1/sqrt(sq/N+s.norm_eps);
            if(n<N) {ws.bc[(ti*2*R+(kind?0:R)+r)*N+n]=float(inv*(double(w[norm+n])*adj-double(bc[n])*inv*inv*cross/N));
                tp[(kind?ll.bnorm:ll.cnorm)+n]+=float(double(adj)*bc[n]*inv);tp[bias+n]=adj;}}
    }
}
// Opt-in v2: one CTA reuses checkpoint replay for four tokens. Left H
// and right GH halos MUST be scalar-reconstructed from their own chunks:
// affine checkpoints and scalar replay endpoints need not be bitwise equal.
constexpr int flash_backward_t=4,flash_backward_stride=129;
template<int ScratchRanks> struct FlashBackwardShared {
    static_assert(ScratchRanks==1||ScratchRanks==2||ScratchRanks==4||ScratchRanks==8);
    float state[flash_backward_t+1][flash_backward_stride];
    float gh[flash_backward_t+1][flash_backward_stride];
    float dq[flash_backward_t][ScratchRanks*lanes],dk[flash_backward_t][ScratchRanks*lanes];
    float coefficient[3][flash_backward_t][lanes];
    float red[lanes];double dred[lanes];
};
static_assert(sizeof(FlashBackwardShared<1>)==16936);
static_assert(sizeof(FlashBackwardShared<2>)==21032);
static_assert(sizeof(FlashBackwardShared<4>)==29224);
static_assert(sizeof(FlashBackwardShared<8>)==45608);
using FlashBackwardSharedMaximum=FlashBackwardShared<8>;
template<int ScratchRanks> __global__ __launch_bounds__(lanes) void flash_token_backward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> initial,Trace<const float> tr,State<const float> seed,State<float> di,
        BackwardWorkspace ws,float* status) {
    const int ttiles=(s.sequence+flash_backward_t-1)/flash_backward_t;
    const int tbase=(blockIdx.x%ttiles)*flash_backward_t,bh=blockIdx.x/ttiles,h=bh%s.heads,b=bh/s.heads,n=threadIdx.x;
    const int N=s.state_dim,P=s.head_dim,R=s.rank,len=valid[b];
    if(len<0||len>s.sequence||tbase>=len) return; // CTA-uniform, including poison padding.
    const int count=std::min(flash_backward_t,len-tbase),begin=tbase/tile*tile,end=std::min(begin+tile,len);
    __shared__ FlashBackwardShared<ScratchRanks> shared;
    float* red=shared.red;double* dred=shared.dred;
    const auto ll=local_layout(s);
    for(int localt=0;localt<count;++localt) {
        for(int r=0;r<R;++r) {shared.dq[localt][r*lanes+n]=0;shared.dk[localt][r*lanes+n]=0;}
        for(int kind=0;kind<3;++kind) shared.coefficient[kind][localt][n]=0;
    }
    __syncthreads();
    for(int pp=0;pp<P;++pp) {
        if(n<N) {
            const int cell=pp*N+n;
            float value=tr.history[history_at(s,tr,bh,begin/tile,cell)];
            for(int t=begin;t<tbase;++t) {const auto c=cached_coeff(s,tr.coefficients,bh,t);value=c.alpha*value+drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);}
            // Only the FIRST previous-state operand crosses a chunk seam.
            shared.state[0][n]=tbase==begin
                ? (tbase?state_value(s,l,p,w,valid,initial,tr,b,h,tbase-1,pp,n):detail::value(initial.ssm,std::size_t(bh)*P*N+cell))
                : value;
            for(int localt=0;localt<count;++localt) {const int t=tbase+localt;const auto c=cached_coeff(s,tr.coefficients,bh,t);
                value=c.alpha*value+drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);shared.state[localt+1][n]=value;}
            float dh=ws.reverse[history_at(s,tr,bh,begin/tile+1,cell)];
            float next=0;
            // Interior right halo is captured before its alpha update.
            for(int t=end-1;t>=tbase+count;--t) {const float g=dh+injection(s,tr,ws,bh,t,pp,n);
                if(t==tbase+count) next=g;dh=cached_coeff(s,tr.coefficients,bh,t).alpha*g;}
            if(tbase+count<len&&tbase+count==end)
                next=gh_value(s,l,p,w,valid,tr,seed,ws,b,h,tbase+count,pp,n);
            shared.gh[count][n]=next; // Final valid token intentionally uses zero next-GH.
            for(int localt=count-1;localt>=0;--localt) {const int t=tbase+localt;
                const float g=dh+injection(s,tr,ws,bh,t,pp,n);shared.gh[localt][n]=g;
                dh=cached_coeff(s,tr.coefficients,bh,t).alpha*g;}
        }
        __syncthreads();
        for(int localt=0;localt<count;++localt) {
            const int t=tbase+localt;const auto ti=token(s,bh,t);const auto c=cached_coeff(s,tr.coefficients,bh,t);
            const float* raw=row(s,p,b,t);float* tp=ws.token_parameters+ti*ll.total;
        const float vx=raw[s.inner()+h*P+pp],pv=t?row(s,p,b,t-1)[s.inner()+h*P+pp]:detail::value(initial.v,std::size_t(bh)*P+pp);
        const float gh=n<N?shared.gh[localt][n]:0;
        const float state=n<N?shared.state[localt+1][n]:0;
        const float prevstate=n<N?shared.state[localt][n]:0;
        shared.coefficient[0][localt][n]=fmaf(gh,prevstate,shared.coefficient[0][localt][n]);float dx=0,initial_v=0;
        const float nextgh=t+1<len&&n<N?shared.gh[localt+1][n]:0;
        const auto nextc=t+1<valid[b]?cached_coeff(s,tr.coefficients,bh,t+1):c;
        for(int r=0;r<R;++r) {const int rp=(h*R+r)*P+pp;const float px=s.mimo?w[l.x+rp]:1;
            const float kt=n<N?tr.k[(ti*R+r)*N+n]:0,pk=n<N?(t?tr.k[((ti-1)*R+r)*N+n]:detail::value(initial.k,(std::size_t(bh)*R+r)*N+n)):0;
            const float adj=ws.gy[(ti*R+r)*P+pp];shared.dq[localt][r*lanes+n]+=adj*state;
            shared.coefficient[1][localt][n]+=gh*pk*pv*px;shared.coefficient[2][localt][n]+=gh*kt*vx*px;shared.dk[localt][r*lanes+n]+=gh*c.gamma*vx*px;
            if(t+1<valid[b]) shared.dk[localt][r*lanes+n]+=nextgh*nextc.beta*vx*px;
            dx+=gh*c.gamma*kt*px+nextgh*(t+1<valid[b]?nextc.beta:0)*kt*px;
            const float gx=reduce(n<N?gh*(c.gamma*kt*vx+c.beta*pk*pv):0,red);
            if(!n&&s.mimo) tp[ll.x+r*P+pp]=gx+adj*w[l.d+h]*vx;
            initial_v+=gh*c.beta*pk*px;
            if(!n) {tp[ll.d]+=adj*vx*px;ws.dx[ti*P+pp]+=adj*w[l.d+h]*px;}
            if(!t&&n<N) di.k[(std::size_t(bh)*R+r)*N+n]+=gh*c.beta*pv*px;
        }
        const float dvalue=reduce(n<N?dx:0,red),div=reduce(n<N?initial_v:0,red);
        if(!n) {ws.dx[ti*P+pp]+=dvalue+(t==valid[b]-1?detail::value(seed.v,std::size_t(bh)*P+pp):0);
            if(!t) di.v[std::size_t(bh)*P+pp]=div;}
        }
        // No lane overwrites replay storage while peers finish the last token.
        __syncthreads();
    }
    for(int localt=0;localt<count;++localt) {
        const int t=tbase+localt;const auto ti=token(s,bh,t);const auto c=cached_coeff(s,tr.coefficients,bh,t);
        const float* raw=row(s,p,b,t);float* tp=ws.token_parameters+ti*ll.total;
        float* dqr=shared.dq[localt];float* dkr=shared.dk[localt];float dq[8]{},dk[8]{};
        for(int r=0;r<R;++r) {dq[r]=dqr[r*lanes+n];dk[r]=dkr[r*lanes+n];}
        const float da=shared.coefficient[0][localt][n],db=shared.coefficient[1][localt][n],dc=shared.coefficient[2][localt][n];
    if(t==valid[b]-1&&n<N) for(int r=0;r<R;++r) dk[r]+=detail::value(seed.k,(std::size_t(bh)*R+r)*N+n);
    const float dda=reduce(n<N?da:0,red),ddb=reduce(n<N?db:0,red),ddc=reduce(n<N?dc:0,red);
    if(!n) {const float dadt=(dda+ddb*(1-c.lam)*c.dt)*c.alpha;
        ws.ddt[ti]=ddb*(1-c.lam)*c.alpha+ddc*c.lam+c.a*dadt;
        ws.da[ti]=c.act>float(s.a_floor)?-dadt*c.dt*(raw[2*s.inner()+2*s.bc()+s.heads+h]>=0?1:c.act*c.act):0;
        ws.dtrap[ti]=(-ddb*c.dt*c.alpha+ddc*c.dt)*c.lam*(1-c.lam);}
    for(int r=0;r<R;++r) {dqr[r*lanes+n]=dq[r];dkr[r*lanes+n]=dk[r];}__syncthreads();
    for(int a=0;a<s.rotary_pairs;++a) {const int i=detail::coord(s,a,false),j=detail::coord(s,a,true);float gp=0;
        if(n==i) for(int r=0;r<R;++r) {const float* q=tr.q+(ti*R+r)*N,*k=tr.k+(ti*R+r)*N;
            gp+=-dqr[r*lanes+i]*q[j]+dqr[r*lanes+j]*q[i]-dkr[r*lanes+i]*k[j]+dkr[r*lanes+j]*k[i];}
        if(n==i) ws.phase[ti*s.rotary_pairs+a]=gp;
    }
    for(int r=0;r<R;++r) {
        float aq=dq[r],ak=dk[r];
        for(int a=0;a<s.rotary_pairs;++a) {const int i=detail::coord(s,a,false),j=detail::coord(s,a,true);
            if(n==i||n==j) {const float phase=tr.phase[ti*s.rotary_pairs+a],co=cosf(phase),sn=sinf(phase);
                aq=n==i?dqr[r*lanes+i]*co+dqr[r*lanes+j]*sn:-dqr[r*lanes+i]*sn+dqr[r*lanes+j]*co;
                ak=n==i?dkr[r*lanes+i]*co+dkr[r*lanes+j]*sn:-dkr[r*lanes+i]*sn+dkr[r*lanes+j]*co;}}
        for(int kind=0;kind<2;++kind) {const float* bc=raw+2*s.inner()+(kind?0:s.bc())+(r*s.groups+h/(s.heads/s.groups))*N;
            const float adj=kind?ak:aq;const auto norm=kind?l.bnorm:l.cnorm,bias=(kind?ll.bbias:ll.cbias)+r*N;
            const double sq=reduce_double(n<N?double(bc[n])*bc[n]:0,dred);
            const double cross=reduce_double(n<N?double(bc[n])*w[norm+n]*adj:0,dred),inv=1/sqrt(sq/N+s.norm_eps);
            if(n<N) {ws.bc[(ti*2*R+(kind?0:R)+r)*N+n]=float(inv*(double(w[norm+n])*adj-double(bc[n])*inv*inv*cross/N));
                tp[(kind?ll.bnorm:ll.cnorm)+n]+=float(double(adj)*bc[n]*inv);tp[bias+n]=adj;}}
    }
        __syncthreads();
    }
}
__global__ void phase_backward_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        State<const float> seed,State<float> di,BackwardWorkspace ws,float* status) {
    const int a=blockIdx.x%s.rotary_pairs,bh=blockIdx.x/s.rotary_pairs,h=bh%s.heads,b=bh/s.heads,lane=threadIdx.x;
    __shared__ float aa[tile],vv[tile];float carry=detail::value(seed.phase,std::size_t(bh)*s.rotary_pairs+a);
    for(int end=valid[b];end>0;end-=tile) {const int t=end-1-lane;float v=t>=0?ws.phase[token(s,bh,t)*s.rotary_pairs+a]:0,one=1;
        affine_scan(one,v,aa,vv,lane,0);if(t>=0) ws.phase[token(s,bh,t)*s.rotary_pairs+a]=v+carry;
        __syncthreads();carry+=vv[std::min(tile,end)-1];__syncthreads();}
    if(!lane) {di.phase[std::size_t(bh)*s.rotary_pairs+a]=carry;if(!isfinite(carry)) fail(status,b);}
}
__global__ void gather_projection_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
        Trace<const float> tr,BackwardWorkspace ws,float* dx,float* status) {
    const auto pos=std::size_t(blockIdx.x)*lanes+threadIdx.x,total=std::size_t(s.batch)*s.sequence*s.width();if(pos>=total) return;
    const int i=int(pos%s.width()),t=int(pos/s.width()%s.sequence),b=int(pos/(std::size_t(s.width())*s.sequence));if(t>=valid[b]) return;
    const int I=s.inner(),off=2*I+2*s.bc();float v=0;
    if(i<2*I) {const int z=i<I?i:i-I,h=z/s.head_dim,pp=z%s.head_dim;const auto ti=token(s,b*s.heads+h,t);v=(i<I?ws.dz:ws.dx)[ti*s.head_dim+pp];}
    else if(i<off) {const int coord=(i-2*I)%s.bc(),r=coord/(s.groups*s.state_dim),g=coord/s.state_dim%s.groups,n=coord%s.state_dim,kind=i<2*I+s.bc()?0:1;
        for(int h=g*(s.heads/s.groups);h<(g+1)*(s.heads/s.groups);++h) v+=ws.bc[(token(s,b*s.heads+h,t)*2*s.rank+kind*s.rank+r)*s.state_dim+n];}
    else if(i<off+3*s.heads) {const int kind=(i-off)/s.heads,h=(i-off)%s.heads;const auto ti=token(s,b*s.heads+h,t);
        if(kind==1) v=ws.da[ti];else if(kind==2) v=ws.dtrap[ti];else {float ddt=ws.ddt[ti];const float* raw=row(s,p,b,t);
            for(int a=0;a<s.rotary_pairs;++a) ddt+=ws.phase[ti*s.rotary_pairs+a]*float(detail::pi)*tanhf(raw[s.width()-s.rotary_pairs+a]);
            v=ddt*detail::sigmoid(raw[off+h]+w[l.dt+h]);}}
    else {const int a=i-(off+3*s.heads);const float* raw=row(s,p,b,t),angle=tanhf(raw[i]);
        for(int h=0;h<s.heads;++h) v+=ws.phase[token(s,b*s.heads+h,t)*s.rotary_pairs+a]*float(detail::pi)*(1-angle*angle)*cached_coeff(s,tr.coefficients,b*s.heads+h,t).dt;}
    dx[pos]=v;if(!isfinite(v)) fail(status,b);
}
__global__ void gather_parameters_kernel(Shape s,Layout l,const int* valid,BackwardWorkspace ws,const float* dx,float* partial,float* status) {
    const auto pos=std::size_t(blockIdx.x)*lanes+threadIdx.x;if(pos>=std::size_t(s.batch)*l.total) return;
    const int b=int(pos/l.total);const auto i=pos%l.total;const auto ll=local_layout(s);float sum=0;
    int h0=0,h1=s.heads;std::size_t local=0;
    if(i<l.dt) local=i; // shared B/C norms
    else if(i<l.bbias) {h0=int(i-l.dt);h1=h0+1;local=ll.dt;}
    else if(i<l.cbias) {const auto z=i-l.bbias;h0=int(z/(s.rank*s.state_dim));h1=h0+1;local=ll.bbias+z%(s.rank*s.state_dim);}
    else if(i<l.d) {const auto z=i-l.cbias;h0=int(z/(s.rank*s.state_dim));h1=h0+1;local=ll.cbias+z%(s.rank*s.state_dim);}
    else if(i<l.x) {h0=int(i-l.d);h1=h0+1;local=ll.d;}
    else if(s.mimo&&i<l.norm) {const auto field=i<l.z?l.x:(i<l.o?l.z:l.o),lf=i<l.z?ll.x:(i<l.o?ll.z:ll.o),z=i-field;
        h0=int(z/(s.rank*s.head_dim));h1=h0+1;local=lf+z%(s.rank*s.head_dim);}
    else {const auto z=i-l.norm;h0=int(z/s.head_dim);h1=h0+1;local=ll.norm+z%s.head_dim;}
    for(int h=h0;h<h1;++h) for(int t=0;t<valid[b];++t) {
        if(i>=l.dt&&i<l.bbias) sum+=dx[(std::size_t(b)*s.sequence+t)*s.width()+2*s.inner()+2*s.bc()+h];
        else sum+=ws.token_parameters[token(s,b*s.heads+h,t)*ll.total+local];}
    partial[pos]=sum;if(!isfinite(sum)) fail(status,b);
}
__global__ void empty_backward_kernel(Shape s,const int* valid,State<const float> seed,State<float> di,float* status) {
    const int bh=blockIdx.x,b=bh/s.heads,n=threadIdx.x;
    if(n<s.head_dim) {const float v=detail::value(seed.v,std::size_t(bh)*s.head_dim+n);if(!isfinite(v)) fail(status,b,2);
        if(!valid[b]) di.v[std::size_t(bh)*s.head_dim+n]=v;}
    for(int r=0;r<s.rank;++r) if(n<s.state_dim) {const float k=detail::value(seed.k,(std::size_t(bh)*s.rank+r)*s.state_dim+n);if(!isfinite(k)) fail(status,b,2);
        if(!valid[b]) di.k[(std::size_t(bh)*s.rank+r)*s.state_dim+n]=k;}
}
}
