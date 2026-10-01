#include "gpu_attention_training.h"
#include "bitlinear.h"
#include "gpu_execution.h"
#include "optimizer_runtime_policy.h"
#include "cuda/device_buffer.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <thread>
#include <utility>
#include <atomic>

namespace nsos::attention_training {
namespace ar=attention_rdna;
namespace {std::array<std::atomic<std::uint64_t>,3> counts{};}
std::array<std::uint64_t,3> dispatch_counters(){return {counts[0].load(),counts[1].load(),counts[2].load()};}
Policy parse_policy(const std::string& v) {
    if(v.empty()||v=="fp32") return Policy::FP32;
    if(v=="rdna_bf16_v1") return Policy::RdnaBF16;
    if(v=="rdna_fp16_v1") return Policy::RdnaFP16;
    throw std::invalid_argument("NSOS_ATTN_TRAINING_PROVIDER must be fp32, rdna_bf16_v1 or rdna_fp16_v1");
}
Policy policy() {
    const char* v=std::getenv("NSOS_ATTN_TRAINING_PROVIDER");
    const auto p=parse_policy(v?v:"");
    if(p!=Policy::FP32&&(optimizer_policy::parse_optimizer_boolean("NSOS_ATTN_TILED_TRAINING",false)||
        optimizer_policy::parse_optimizer_boolean("NSOS_ATTN_BWD_HOST",false)))
        throw std::invalid_argument("RDNA attention conflicts with FP32 tiled training/host backward");
    return p;
}
const char* policy_identity(Policy p) {
    switch(p) {
    case Policy::FP32:return "fp32_existing_v1";
    case Policy::RdnaBF16:return "rdna_bf16_v1";
    case Policy::RdnaFP16:return "rdna_fp16_v1";
    }
    throw std::invalid_argument("invalid attention policy");
}
namespace {
void tensor(const Tensor& x,const std::vector<int>& dims,const char* name) {
    std::size_t count=1;
    if(x.get_device()!=Device::GPU||x.shape.dims!=dims||x.shape.strides.size()!=dims.size())
        throw std::invalid_argument(std::string("RDNA attention tensor shape/device: ")+name);
    for(int i=static_cast<int>(dims.size())-1;i>=0;--i) {
        if(dims[i]<=0||x.shape.strides[i]!=count||count>SIZE_MAX/static_cast<std::size_t>(dims[i]))
            throw std::invalid_argument(std::string("RDNA attention noncontiguous tensor: ")+name);
        count*=dims[i];
    }
    if(x.size!=count||!x.raw_data()) throw std::invalid_argument("RDNA attention extent/storage");
}
void geometry(const Config& c) {
    if(!ar::geometry_eligible(c.shape)||c.start_position<0||
        c.start_position>INT_MAX-c.shape.sequence||ar::query_elements(c.shape)>INT_MAX/2)
        throw std::invalid_argument("RDNA attention integration geometry/extent rejected");
}
std::vector<int> prefixes(const Config& c,const std::vector<int>& v) {
    auto result=v.empty()?std::vector<int>(c.shape.batch,c.shape.sequence):v;
    if(result.size()!=static_cast<std::size_t>(c.shape.batch)) throw std::invalid_argument("RDNA attention prefix batch extent");
    for(int n:result) if(n<0||n>c.shape.sequence) throw std::invalid_argument("RDNA attention prefix range");
    return result;
}
void enqueued(bool ok,const char* where) {
    if(!ok) throw std::runtime_error(std::string("RDNA attention enqueue failed: ")+where+"; discard attempt and drain owning lane");
}
#ifdef USE_CUDA
void check(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
struct Lane {
    int device=-1; cudaStream_t stream=nullptr;
    std::thread::id owner=std::this_thread::get_id();
    const gpu::ExecutionContext* context=nullptr;
    Lane() {
        check(cudaGetDevice(&device),"Attention owning device");
        stream=gpu::current_stream();context=&gpu::current_execution_context();
    }
    void require() const {
        int d=-1;check(cudaGetDevice(&d),"Attention current device");
        if(d!=device||stream!=gpu::current_stream()||owner!=std::this_thread::get_id()||
            context!=&gpu::current_execution_context())
            throw std::runtime_error("RDNA attention requires its owning thread/device/stream/context");
    }
    void require_owner() const {
        int d=-1;check(cudaGetDevice(&d),"Attention cancellation device");
        if(d!=device||owner!=std::this_thread::get_id())
            throw std::runtime_error("RDNA attention cancellation crossed thread/device");
    }
    void drain() const noexcept {
        int previous=-1;
        if(cudaGetDevice(&previous)!=cudaSuccess||cudaSetDevice(device)!=cudaSuccess) std::terminate();
        // Fail-stop on a broken completion boundary: releasing referenced Tensor
        // storage after a failed fence would be a use-after-free on device.
        if(cudaStreamSynchronize(stream)!=cudaSuccess) std::terminate();
        record_gpu_stream_synchronization();
        if(previous!=device&&cudaSetDevice(previous)!=cudaSuccess) std::terminate();
    }
};
struct Ledger {
    Lane lane;cuda_detail::DeviceBuffer<int> issue;
    cudaEvent_t completion=nullptr;
    std::vector<Parameter*> parameters;
    explicit Ledger(std::vector<Parameter*> p):parameters(std::move(p)) {
        if(!issue.ensure(1)) throw std::runtime_error("Attention status allocation");
        lane.require();check(cudaEventCreate(&completion),"Attention status event");
        check(cudaMemsetAsync(issue.get(),0,sizeof(int),lane.stream),"Attention initial status");
        record();
    }
    ~Ledger(){
        lane.drain();
        if(completion) {
            int prior=-1;
            if(cudaGetDevice(&prior)!=cudaSuccess||cudaSetDevice(lane.device)!=cudaSuccess||
                cudaEventSynchronize(completion)!=cudaSuccess||cudaEventDestroy(completion)!=cudaSuccess)
                std::terminate();
            if(prior!=lane.device&&cudaSetDevice(prior)!=cudaSuccess) std::terminate();
        }
    }
    // Trainer operates outside Jamba's bound ExecutionContext. Scope enter/exit
    // events order model tensors; this event independently orders ledger uses.
    void wait() {
        int d=-1;check(cudaGetDevice(&d),"Attention ledger current device");
        if(d!=lane.device||lane.owner!=std::this_thread::get_id())
            throw std::runtime_error("Attention ledger crossed thread/device");
        check(cudaStreamWaitEvent(gpu::current_stream(),completion,0),"Attention ledger dependency");
    }
    void record(){check(cudaEventRecord(completion,gpu::current_stream()),"Attention ledger completion");}
    bool matches(const std::vector<Parameter*>& p) const {
        for(auto* a:parameters) if(std::find(p.begin(),p.end(),a)!=p.end()) return true;
        return false;
    }
};
thread_local std::vector<std::weak_ptr<Ledger>> ledgers;
template<class F> void visit(const std::vector<Parameter*>& params,F action) {
    for(auto it=ledgers.begin();it!=ledgers.end();) {
        if(auto p=it->lock()) {
            if(p->matches(params)) {p->wait();action(*p);p->record();}
            ++it;
        } else it=ledgers.erase(it);
    }
}
#endif
}
struct Tape::Impl {
    Config config;
    Tensor source_q,source_kv,cs,sn,q,k,v,out,mx,inv,dy,delta,dq,dk,dv;
    ProjectionGradient result;
    bool consumed=false;
#ifdef USE_CUDA
    Lane lane;
    cuda_detail::DeviceBuffer<int> valid,status;
    ~Impl(){lane.drain();}
#endif
};
Tape::Tape(std::unique_ptr<Impl> p):impl_(std::move(p)){}
Tape::~Tape()=default;
std::unique_ptr<Tape> Tape::forward(const Config& c,const Tensor& q,const Tensor& kv,
    const Tensor& cs,const Tensor& sn,const std::vector<int>& v,std::unique_ptr<Tape> reuse) {
    geometry(c);const auto& s=c.shape;
    const int qdim=s.query_heads*s.head_dim,kdim=s.kv_heads*s.head_dim;
    tensor(q,{s.batch,s.sequence,qdim},"Q projection");tensor(kv,{s.batch,s.sequence,2*kdim},"KV projection");
    if(cs.shape.size()!=2||cs.shape[0]<c.start_position+s.sequence)
        throw std::invalid_argument("RDNA attention rotary table capacity");
    // D=1 has no rotation; accept one inert table column for valid Tensor ABI.
    tensor(cs,{cs.shape[0],std::max(1,s.head_dim/2)},"RoPE cosine");tensor(sn,cs.shape.dims,"RoPE sine");
    auto valid=prefixes(c,v);
#ifdef USE_CUDA
    if(!ar::supported(s)) throw std::runtime_error("Requested RDNA attention lacks compiled gfx11 wave32/rocWMMA capability");
    std::unique_ptr<Impl> p;
    if(reuse) {
        reuse->impl_->lane.require();
        if(!reuse->consumed()) throw std::runtime_error("Attention reuse requires a consumed/cancelled tape");
        if(reuse->impl_->source_q.shape==q.shape&&reuse->impl_->source_kv.shape==kv.shape&&reuse->impl_->cs.shape==cs.shape&&
            reuse->impl_->config.shape.query_heads==s.query_heads&&reuse->impl_->config.shape.kv_heads==s.kv_heads&&
            reuse->impl_->config.shape.head_dim==s.head_dim)
            p=std::move(reuse->impl_);
        reuse.reset();
    }
    const bool recycled=bool(p);
    if(!p) p=std::make_unique<Impl>();p->config=c;p->consumed=false;
    if(recycled) {
        p->source_q.copy_from(q);p->source_kv.copy_from(kv);p->cs.copy_from(cs);p->sn.copy_from(sn);
    } else {
        p->source_q=q.clone();p->source_kv=kv.clone();p->cs=cs.clone();p->sn=sn.clone();
    }
    const std::vector<int> qs{s.batch,s.sequence,s.query_heads,s.head_dim},ks{s.batch,s.sequence,s.kv_heads,s.head_dim};
    if(!recycled) {
        p->q=Tensor::uninitialized(qs,Device::GPU);p->k=Tensor::uninitialized(ks,Device::GPU);p->v=Tensor::uninitialized(ks,Device::GPU);
        p->out=Tensor::uninitialized(qs,Device::GPU);
        p->mx=Tensor::uninitialized({s.batch,s.sequence,s.query_heads},Device::GPU);
        p->inv=Tensor::uninitialized(p->mx.shape.dims,Device::GPU);
    }
    if(!p->valid.ensure(s.batch)||!p->status.ensure(s.batch)) throw std::runtime_error("RDNA tape metadata allocation");
    p->lane.require();
    check(cudaMemcpy(p->valid.get(),valid.data(),valid.size()*sizeof(int),cudaMemcpyHostToDevice),"Attention prefix upload");
    record_gpu_transfer(Device::GPU,Device::CPU,valid.size()*sizeof(int));
    enqueued(ar::prepare_projected(s,c.start_position,cs.shape[0],p->source_q.raw_data(),p->source_kv.raw_data(),
        p->cs.raw_data(),p->sn.raw_data(),p->valid.get(),p->q.raw_data(),p->k.raw_data(),p->v.raw_data()),"split/RoPE");
    enqueued(ar::forward(s,p->q.raw_data(),p->k.raw_data(),p->v.raw_data(),p->valid.get(),p->out.raw_data(),p->mx.raw_data(),p->inv.raw_data(),p->status.get()),"forward");
    enqueued(ar::inspect_result(s,qdim,p->valid.get(),p->out.raw_data(),p->status.get()),"forward finite status");
    ++counts[0];
    return std::unique_ptr<Tape>(new Tape(std::move(p)));
#else
    throw std::runtime_error("RDNA attention requires a GPU build; no CPU fallback");
#endif
}
Tensor Tape::output() const {
#ifdef USE_CUDA
    impl_->lane.require();return impl_->out.reshape(impl_->source_q.shape.dims).clone();
#else
    throw std::runtime_error("GPU required");
#endif
}
ProjectionGradient Tape::backward(const Tensor& dy) {
    if(impl_->consumed) throw std::runtime_error("RDNA attention tape consumed/cancelled");
    const auto& s=impl_->config.shape;tensor(dy,impl_->source_q.shape.dims,"dO");
#ifdef USE_CUDA
    auto& p=*impl_;p.lane.require();p.consumed=true;
    if(p.dy.size) p.dy.copy_from(dy);else p.dy=dy.clone();
    if(!p.delta.size) {
        p.delta=Tensor::uninitialized(p.mx.shape.dims,Device::GPU);
        p.dq=Tensor::uninitialized(p.q.shape.dims,Device::GPU);p.dk=Tensor::uninitialized(p.k.shape.dims,Device::GPU);p.dv=Tensor::uninitialized(p.v.shape.dims,Device::GPU);
        p.result={Tensor::uninitialized(p.source_q.shape.dims,Device::GPU),Tensor::uninitialized(p.source_kv.shape.dims,Device::GPU)};
    }
    enqueued(ar::backward(s,p.q.raw_data(),p.k.raw_data(),p.v.raw_data(),p.out.raw_data(),p.dy.raw_data(),p.valid.get(),
        p.mx.raw_data(),p.inv.raw_data(),p.status.get(),p.delta.raw_data(),p.dq.raw_data(),p.dk.raw_data(),p.dv.raw_data()),"backward");
    enqueued(ar::join_projected_gradient(s,p.config.start_position,p.cs.shape[0],p.dq.raw_data(),p.dk.raw_data(),p.dv.raw_data(),
        p.cs.raw_data(),p.sn.raw_data(),p.valid.get(),p.status.get(),p.result.q.raw_data(),p.result.kv.raw_data()),"RoPE VJP/KV concat");
    enqueued(ar::inspect_result(s,s.query_heads*s.head_dim,p.valid.get(),p.result.q.raw_data(),p.status.get()),"dQ status");
    enqueued(ar::inspect_result(s,2*s.kv_heads*s.head_dim,p.valid.get(),p.result.kv.raw_data(),p.status.get()),"dKV status");
    // Returned grads can be mutated; they never participate in another backward.
    ++counts[1];return {p.result.q.clone(),p.result.kv.clone()};
#else
    throw std::runtime_error("GPU required");
#endif
}
const int* Tape::device_status() const {
#ifdef USE_CUDA
    impl_->lane.require();return impl_->status.get();
#else
    return nullptr;
#endif
}
std::vector<int> Tape::audit_status() const {
#ifdef USE_CUDA
    impl_->lane.require();check(cudaStreamSynchronize(impl_->lane.stream),"Attention audit fence");
    record_gpu_stream_synchronization();std::vector<int> result(impl_->config.shape.batch);
    check(cudaMemcpy(result.data(),impl_->status.get(),result.size()*sizeof(int),cudaMemcpyDeviceToHost),"Attention audit status");
    record_gpu_transfer(Device::CPU,Device::GPU,result.size()*sizeof(int));return result;
#else
    throw std::runtime_error("GPU required");
#endif
}
bool Tape::consumed() const {return impl_->consumed;}
void Tape::cancel() {
#ifdef USE_CUDA
    impl_->lane.require_owner(); // host-only invalidation; reset_session has no bound lane
#endif
    impl_->consumed=true;
}
std::size_t Tape::workspace_bytes() const {
    std::size_t n=0;const auto& p=*impl_;
    for(const auto* x:{&p.source_q,&p.source_kv,&p.cs,&p.sn,&p.q,&p.k,&p.v,&p.out,&p.mx,&p.inv,&p.dy,&p.delta,&p.dq,&p.dk,&p.dv,&p.result.q,&p.result.kv}) n+=x->size*sizeof(float);
#ifdef USE_CUDA
    return n+(p.valid.capacity()+p.status.capacity())*sizeof(int);
#else
    return n;
#endif
}
struct Provider::Impl {
    BitLinear *q,*kv,*out;
    std::unique_ptr<Tape> tape;
    Tensor input,attention,dy,result;
    Config config;int original_rank=0,matmul_mode=0;
    bool ready=false;
    Policy selected=Policy::FP32;
    struct Stamp {Parameter* p;std::uint64_t version;const float* storage;bool trainable;};
    std::vector<Stamp> stamps;
#ifdef USE_CUDA
    std::shared_ptr<Ledger> ledger;
    cuda_detail::DeviceBuffer<int> valid;
    void snapshot() {
        stamps.clear();for(auto* l:{q,kv,out}) for(auto* p:l->parameters())
            stamps.push_back({p,p->version,p->data.raw_data(),p->trainable});
        matmul_mode=matmul_precision_mode();
    }
    void unchanged() {
        if(policy()!=selected||matmul_precision_mode()!=matmul_mode)
            throw std::runtime_error("RDNA attention policy changed between forward/backward");
        std::vector<Parameter*> now;for(auto* l:{q,kv,out}) {
            if(!l->training_mode()) throw std::runtime_error("Attention projection left training mode");
            auto p=l->parameters();now.insert(now.end(),p.begin(),p.end());
        }
        if(now.size()!=stamps.size()) throw std::runtime_error("Attention projection parameter registry changed");
        for(std::size_t i=0;i<now.size();++i) {
            const auto& s=stamps[i];
            if(now[i]!=s.p||s.p->version!=s.version||s.p->data.raw_data()!=s.storage||s.p->trainable!=s.trainable)
                throw std::runtime_error("Attention projection parameter changed before backward");
        }
    }
    void mask(Tensor& x,int features) {enqueued(ar::mask_prefix(config.shape,features,valid.get(),x.raw_data()),"projection prefix mask");}
    void publish() {ledger->wait();enqueued(ar::merge_device_status(tape->device_status(),config.shape.batch,ledger->issue.get()),"sticky status");ledger->record();}
    ~Impl(){if(ledger) ledger->lane.drain();}
#endif
};
Provider::Provider(BitLinear& q,BitLinear& kv,BitLinear& out):impl_(std::make_unique<Impl>()) {
    impl_->q=&q;impl_->kv=&kv;impl_->out=&out;
    if(&q==&kv||&q==&out||&kv==&out) throw std::invalid_argument("Attention projections require distinct BitLinear owners");
}
Provider::~Provider()=default;
Tensor Provider::forward(const Tensor& input,const Config& c,const Tensor& cs,const Tensor& sn,const std::vector<int>& lengths) {
    geometry(c);auto& p=*impl_;const auto& s=c.shape;
    const int dim=s.query_heads*s.head_dim,kdim=s.kv_heads*s.head_dim;
    if(pending()) throw std::runtime_error("RDNA attention outstanding forward requires backward/cancel");
    p.selected=policy();
    if(p.selected==Policy::FP32||s.precision!=(p.selected==Policy::RdnaBF16?ar::Precision::BF16:ar::Precision::FP16))
        throw std::invalid_argument("Attention provider/precision selection mismatch");
#ifdef USE_CUDA
    if(!ar::supported(s)) throw std::runtime_error("Requested RDNA attention unavailable; no fallback");
#else
    throw std::runtime_error("Requested RDNA attention requires a GPU build; no fallback");
#endif
    if(p.q->input_features()!=dim||p.q->output_features()!=dim||p.kv->input_features()!=dim||
        p.kv->output_features()!=2*kdim||p.out->input_features()!=dim||p.out->output_features()!=dim)
        throw std::invalid_argument("Attention projection geometry mismatch");
    for(auto* l:{p.q,p.kv,p.out}) if(!l->training_mode()) throw std::invalid_argument("Attention projections require training mode");
    bool participating=false;
    for(auto* l:{p.q,p.kv,p.out}) for(auto* a:l->parameters()) participating|=a->trainable;
    if(!participating)
        throw std::invalid_argument("RDNA attention requires a trainable projection parameter to bind the Trainer status cohort");
    p.original_rank=input.shape.size();
    std::vector<int> dims;
    if(p.original_rank==1&&s.batch==1&&s.sequence==1) dims={dim};
    else if(p.original_rank==2&&s.batch==1) dims={s.sequence,dim};
    else if(p.original_rank==3) dims={s.batch,s.sequence,dim};
    else throw std::invalid_argument("Attention input rank/batch contract");
    tensor(input,dims,"projection input");auto valid=prefixes(c,lengths);
#ifdef USE_CUDA
    p.config=c;
    if(!p.ledger) {
        std::vector<Parameter*> params;for(auto* l:{p.q,p.kv,p.out}) {
            auto v=l->parameters();params.insert(params.end(),v.begin(),v.end());
        }
        p.ledger=std::make_shared<Ledger>(std::move(params));ledgers.push_back(p.ledger);
    }
    p.ledger->lane.require();p.snapshot();
    p.ledger->parameters.clear();for(const auto& stamp:p.stamps)p.ledger->parameters.push_back(stamp.p);
    if(!p.valid.ensure(s.batch)) throw std::runtime_error("Attention provider valid allocation");
    p.ledger->lane.require();check(cudaMemcpy(p.valid.get(),valid.data(),valid.size()*sizeof(int),cudaMemcpyHostToDevice),"Attention provider prefixes");
    record_gpu_transfer(Device::GPU,Device::CPU,valid.size()*sizeof(int));
    p.input=input.reshape({s.batch,s.sequence,dim}).clone();p.mask(p.input,dim);
    Tensor q=p.q->forward(p.input),kv=p.kv->forward(p.input);
    p.tape=Tape::forward(c,q,kv,cs,sn,valid,std::move(p.tape));p.publish();p.attention=p.tape->output();
    Tensor result=p.out->forward(p.attention);
    p.mask(result,dim);
    enqueued(ar::inspect_result(s,dim,p.valid.get(),result.raw_data(),const_cast<int*>(p.tape->device_status())),"output projection status");p.publish();
    p.ready=true;return result.reshape(dims);
#else
    throw std::runtime_error("GPU required");
#endif
}
Tensor Provider::backward(const Tensor& dy) {
    auto& p=*impl_;if(!pending()) throw std::runtime_error("Attention backward without pending tape");
#ifdef USE_CUDA
    p.ledger->lane.require();p.unchanged();const auto& s=p.config.shape;const int dim=s.query_heads*s.head_dim;
    std::vector<int> dims=p.original_rank==1?std::vector<int>{dim}:p.original_rank==2?std::vector<int>{s.sequence,dim}:std::vector<int>{s.batch,s.sequence,dim};
    tensor(dy,dims,"projection output adjoint");
    p.ready=false; // any partial projection backward prohibits a second attempt
    p.dy=dy.reshape({s.batch,s.sequence,dim}).clone();p.mask(p.dy,dim);
    // Replay restores per-BitLinear activation/QAT state without gradients.
    (void)p.out->forward(p.attention);
    Tensor go=p.out->backward(p.dy);
    auto g=p.tape->backward(go);p.publish();
    (void)p.q->forward(p.input);Tensor dxq=p.q->backward(g.q);
    (void)p.kv->forward(p.input);Tensor dxkv=p.kv->backward(g.kv);
    p.result=dxq.add(dxkv);p.mask(p.result,dim);
    enqueued(ar::inspect_result(s,dim,p.valid.get(),p.result.raw_data(),const_cast<int*>(p.tape->device_status())),"input gradient status");p.publish();
    return p.result.reshape(dims);
#else
    throw std::runtime_error("GPU required");
#endif
}
void Provider::cancel_pending(){impl_->ready=false;if(impl_->tape) impl_->tape->cancel();}
bool Provider::pending() const {return impl_->ready&&impl_->tape&&!impl_->tape->consumed();}
std::size_t Provider::workspace_bytes() const {
    const auto& p=*impl_;auto n=(p.tape?p.tape->workspace_bytes():0)+(p.input.size+p.attention.size+p.dy.size+p.result.size)*sizeof(float);
#ifdef USE_CUDA
    n+=(p.valid.capacity()+(p.ledger?p.ledger->issue.capacity():0))*sizeof(int);
#endif
    return n;
}
void reset_status(const std::vector<Parameter*>& params) {
#ifdef USE_CUDA
    visit(params,[](Ledger& p){check(cudaMemsetAsync(p.issue.get(),0,sizeof(int),gpu::current_stream()),"Attention group status reset");});
#endif
}
void merge_status(const std::vector<Parameter*>& params,int* issue) {
#ifdef USE_CUDA
    visit(params,[&](Ledger& p){enqueued(ar::merge_device_status(p.issue.get(),1,issue),"trainer finite gate");++counts[2];});
#endif
}
}
