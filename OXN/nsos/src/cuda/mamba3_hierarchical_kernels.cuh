#pragma once
// Independent opt-in chunk tree. Include after mamba3_parallel_kernels.cuh.
// No phase scan lives here. New association is intentionally versioned.
namespace nsos::mamba3_block::hierarchical_detail {
namespace pd=parallel_detail;
constexpr int arity=32;
__device__ inline void pair_scan(float& a,float& v,float* aa,float* vv,int lane,int base) {
    aa[threadIdx.x]=a;vv[threadIdx.x]=v;__syncthreads();
    for(int offset=1;offset<arity;offset*=2) {
        float na=a,nv=v;
        if(lane>=offset) {na=a*aa[base+lane-offset];nv=fmaf(a,vv[base+lane-offset],v);}
        __syncthreads();a=na;v=nv;aa[threadIdx.x]=a;vv[threadIdx.x]=v;__syncthreads();
    }
}
// Four independent PN cells/chunks per CTA. Each chunk has no carry from
// another chunk: its affine summary is computed from the local token scan.
// Reverse leaves are stored in reverse chunk order, including empty tails.
__global__ void summary_kernel(Shape s,Layout l,const float* p,const float* w,const int* valid,
    State<const float> initial,Trace<const float> tr,BackwardWorkspace ws,bool reverse,
    float* aa_out,float* bb_out,float* status) {
    const int Q=(s.sequence+pd::tile-1)/pd::tile,lane=threadIdx.x%pd::tile,base=threadIdx.x-lane;
    const auto warp=std::size_t(blockIdx.x)*4+threadIdx.x/pd::tile;
    const auto id=warp/Q;const int q=int(warp%Q);
    const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const int cell=int(id%(s.head_dim*s.state_dim)),n=cell%s.state_dim,pp=cell/s.state_dim;
    const auto bh=id/(s.head_dim*s.state_dim);const int b=int(bh/s.heads),h=int(bh%s.heads);
    const int len=id<cells?valid[b]:0,chunk=reverse?Q-1-q:q,begin=chunk*pd::tile;
    const int t=reverse?std::min(begin+pd::tile,len)-1-lane:begin+lane;
    float a=1,v=0;
    if(id<cells&&len>=0&&len<=s.sequence&&t>=begin&&t<len) {
        const auto c=pd::cached_coeff(s,tr.coefficients,int(bh),t);a=c.alpha;
        v=reverse?a*pd::injection(s,tr,ws,int(bh),t,pp,n)
                 :pd::drive(s,l,p,w,tr,initial,b,h,t,pp,n,c);
    }
    __shared__ float aa[pd::lanes],vv[pd::lanes];
    pd::affine_scan(a,v,aa,vv,lane,base);
    if(lane==pd::tile-1&&id<cells) {
        const auto index=id*Q+q;aa_out[index]=a;bb_out[index]=v;
        if(len<0||len>s.sequence)pd::fail(status,b,1);
        else if(!isfinite(a)||!isfinite(v))pd::fail(status,b);
    }
}
// Scan each group independently, emit a summary into the next level. All
// padded warps/lanes participate in barriers; padding never reads source.
__global__ void level_kernel(Shape s,int length,std::size_t offset,std::size_t next_offset,
    bool emit_next,float* tree_a,float* tree_b,float* status) {
    const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const int groups=(length+arity-1)/arity,lane=threadIdx.x%arity,base=threadIdx.x-lane;
    const auto warp=std::size_t(blockIdx.x)*4+threadIdx.x/arity,id=warp/groups;
    const int group=int(warp%groups),q=group*arity+lane;
    const auto i=offset+id*length+q;float a=1,v=0;
    if(id<cells&&q<length) {a=tree_a[i];v=tree_b[i];}
    __shared__ float aa[pd::lanes],vv[pd::lanes];pair_scan(a,v,aa,vv,lane,base);
    if(id<cells&&q<length) {tree_a[i]=a;tree_b[i]=v;
        if(!isfinite(a)||!isfinite(v))pd::fail(status,int(id/(s.head_dim*s.state_dim*s.heads)));}
    if(emit_next&&lane==arity-1&&id<cells) {
        const auto j=next_offset+id*groups+group;tree_a[j]=a;tree_b[j]=v;
    }
}
// Parent already has globally inclusive pairs. Previous parent group supplies
// the exclusive carry, applied to every independent child pair with fmaf.
__global__ void fixup_kernel(Shape s,int length,std::size_t offset,std::size_t parent_offset,
    float* tree_a,float* tree_b,float* status) {
    const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const auto i=std::size_t(blockIdx.x)*pd::lanes+threadIdx.x;if(i>=cells*length)return;
    const auto id=i/length;const int q=int(i%length),group=q/arity;
    if(!group)return;
    const int parent_length=(length+arity-1)/arity;
    const auto j=parent_offset+id*parent_length+group-1;
    const float a=tree_a[offset+i],v=fmaf(a,tree_b[j],tree_b[offset+i]);
    tree_a[offset+i]=a*tree_a[j];tree_b[offset+i]=v;
    if(!isfinite(tree_a[offset+i])||!isfinite(v))pd::fail(status,int(id/(s.head_dim*s.state_dim*s.heads)));
}
// Chunk boundary slots use one global prefix application to the public initial
// (or final-adjoint) state. They are not scalar replay endpoints. Replay halos
// MUST continue using the existing scalar helpers from their owned chunks.
__global__ void publish_kernel(Shape s,const int* valid,State<const float> seed,
    bool reverse,const float* tree_a,const float* tree_b,float* boundaries,
    float* final_ssm,float* status) {
    const int Q=(s.sequence+pd::tile-1)/pd::tile,slots=Q+1;
    const auto cells=std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim;
    const auto i=std::size_t(blockIdx.x)*pd::lanes+threadIdx.x;if(i>=cells*slots)return;
    const auto id=i/slots,bh=id/(s.head_dim*s.state_dim);
    const int slot=int(i%slots),cell=int(id%(s.head_dim*s.state_dim)),b=int(bh/s.heads);
    const float initial=detail::value(seed.ssm,id);float v=initial;
    const bool base=reverse?slot==Q:slot==0;
    if(!base&&valid[b]>0&&valid[b]<=s.sequence) {
        const int q=reverse?Q-1-slot:slot-1;const auto j=id*Q+q;
        v=fmaf(tree_a[j],initial,tree_b[j]);
    }
    boundaries[(bh*slots+slot)*s.head_dim*s.state_dim+cell]=v;
    if((reverse&&slot==0)||(!reverse&&slot==Q))final_ssm[id]=v;
    if(!isfinite(initial))pd::fail(status,b,2);
    else if(!isfinite(v))pd::fail(status,b);
}
}
