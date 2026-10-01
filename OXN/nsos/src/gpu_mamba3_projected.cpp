#include "gpu_mamba3_projected.h"
#include "cuda/device_buffer.h"
#include "gpu_execution.h"
#include <climits>
#include <thread>

namespace nsos::mamba3_projected {
namespace {
namespace pg=mamba3_preprocess_gpu;
void shape(const Tensor& t,const std::vector<int>& dims,const char* name) {
    if (t.get_device()!=Device::GPU||t.shape.dims!=dims||t.shape.strides.size()!=dims.size())
        throw std::invalid_argument(std::string("Mamba3 projected tensor shape/device: ")+name);
    std::size_t count=1;
    for (int i=static_cast<int>(dims.size())-1;i>=0;--i) {
        if (dims[i]<=0||t.shape.strides[i]!=count||count>std::numeric_limits<std::size_t>::max()/dims[i])
            throw std::invalid_argument(std::string("Mamba3 projected stride/overflow: ")+name);
        count*=dims[i];
    }
    if (t.size!=count||!t.raw_data()) throw std::invalid_argument(std::string("Mamba3 projected extent/storage: ")+name);
}
void validate(const Shape& s,const Operands& x) {
    shape(x.q,{s.batch,s.sequence,s.groups,s.state_dim},"Q");shape(x.k,x.q.shape.dims,"K");
    shape(x.v,{s.batch,s.sequence,s.heads,s.head_dim},"V");if (x.z.size) shape(x.z,x.v.shape.dims,"Z");
    shape(x.raw_a,{s.batch,s.sequence,s.heads},"raw A");shape(x.raw_dt,x.raw_a.shape.dims,"raw DT");shape(x.trap,x.raw_a.shape.dims,"Trap");
    shape(x.angles,{s.batch,s.sequence,s.rotary_pairs},"shared angles");
    shape(x.q_norm,{s.state_dim},"Q norm");shape(x.k_norm,{s.state_dim},"K norm");shape(x.dt_bias,{s.heads},"DT bias");
    shape(x.q_bias,{s.heads,s.state_dim},"Q bias");shape(x.k_bias,x.q_bias.shape.dims,"K bias");if (x.d.size) shape(x.d,{s.heads},"D");
}
void validate_state(const Shape& s,const State& x) {
    if (!(x.phase.size||x.ssm.size||x.k.size||x.v.size)) return;
    shape(x.phase,{s.batch,s.heads,s.rotary_pairs},"state phase");shape(x.ssm,{s.batch,s.heads,s.head_dim,s.state_dim},"state SSM");
    shape(x.k,{s.batch,s.heads,s.state_dim},"state K");shape(x.v,{s.batch,s.heads,s.head_dim},"state V");
}
Operands clone(const Operands& x) {
    auto cp=[](const Tensor& t) {return t.size?t.clone():Tensor();};
    return {cp(x.q),cp(x.k),cp(x.v),cp(x.z),cp(x.raw_a),cp(x.raw_dt),cp(x.trap),cp(x.angles),cp(x.q_norm),cp(x.k_norm),cp(x.dt_bias),cp(x.q_bias),cp(x.k_bias),cp(x.d)};
}
pg::Gradient destination(Operands& x) {
    return {x.q.raw_data(),x.k.raw_data(),x.raw_a.raw_data(),x.raw_dt.raw_data(),x.angles.raw_data(),x.q_norm.raw_data(),x.k_norm.raw_data(),x.dt_bias.raw_data()};
}
#ifdef USE_CUDA
void checked(cudaError_t e,const char* name) {if (e!=cudaSuccess) throw std::runtime_error(std::string(name)+": "+cudaGetErrorString(e));}
#endif
}
struct Tape::Impl {
    Shape shape;Config config;Operands input;Gradients gradients;
    mamba3_siso::Operands prepared;
    Tensor inverse,partials;
    std::unique_ptr<mamba3_siso::Tape> recurrence;
    bool consumed=false;std::thread::id owner=std::this_thread::get_id();
#ifdef USE_CUDA
    int device=-1;cudaStream_t stream=nullptr;
    cuda_detail::DeviceBuffer<int> valid,status;
    void lane() const {
        int current=-1;checked(cudaGetDevice(&current),"Mamba3 projected device");
        if (std::this_thread::get_id()!=owner||current!=device||gpu::current_stream()!=stream)
            throw std::runtime_error("Mamba3 projected tape requires its original thread/device/stream");
    }
    pg::Input source() const {
        return {input.q.raw_data(),input.k.raw_data(),input.raw_a.raw_data(),input.raw_dt.raw_data(),input.angles.raw_data(),
            input.q_norm.raw_data(),input.k_norm.raw_data(),input.dt_bias.raw_data(),valid.get(),inverse.raw_data()};
    }
    void copy_status() {
        checked(cudaMemcpyAsync(status.get(),recurrence->device_status(),shape.batch*sizeof(int),cudaMemcpyDeviceToDevice,stream),"Mamba3 chain status");
        record_gpu_transfer(Device::GPU,Device::GPU,shape.batch*sizeof(int));
    }
#else
    void lane() const {throw std::runtime_error("Mamba3 projected tape requires GPU build");}
#endif
};
Tape::Tape(std::unique_ptr<Impl> p):impl_(std::move(p)) {}
Tape::~Tape()=default;
std::unique_ptr<Tape> Tape::forward(const Shape& s,const Operands& x,const State& initial,const std::vector<int>& lengths,Config c) {
    if (!pg::eligible(s,c)) throw std::invalid_argument("Mamba3 projected unsupported geometry/config");
    validate(s,x);validate_state(s,initial);
    const auto prefixes=lengths.empty()?std::vector<int>(s.batch,s.sequence):lengths;
    if (prefixes.size()!=static_cast<std::size_t>(s.batch)) throw std::invalid_argument("Mamba3 projected prefix shape");
    for (int n:prefixes) if (n<0||n>s.sequence) throw std::invalid_argument("Mamba3 projected invalid prefix");
    if (pg::partial_elements(s)>INT_MAX) throw std::length_error("Mamba3 preprocessing scratch exceeds Tensor flat ABI");
#ifdef USE_CUDA
    if (!mamba3_siso_gpu::supported(s)) throw std::runtime_error("Mamba3 projected requires compiled RDNA3 wave32 capability");
    auto p=std::make_unique<Impl>();p->shape=s;p->config=c;
    checked(cudaGetDevice(&p->device),"Mamba3 projected selected device");p->stream=gpu::current_stream();
    p->input=clone(x);
    auto make=[](const Tensor& t) {return Tensor::uninitialized(t.shape.dims,Device::GPU);};
    p->prepared={make(x.q),make(x.k),p->input.v,p->input.z,make(x.raw_a),make(x.raw_dt),p->input.trap,
        Tensor::uninitialized({s.batch,s.sequence,s.heads,s.rotary_pairs},Device::GPU),p->input.q_bias,p->input.k_bias,p->input.d};
    p->inverse=Tensor::uninitialized({s.batch,s.sequence,s.groups,2},Device::GPU);
    p->partials=Tensor::uninitialized({static_cast<int>(pg::partial_elements(s))},Device::GPU);
    if (!p->valid.ensure(s.batch)||!p->status.ensure(s.batch)) throw std::runtime_error("Mamba3 projected metadata allocation");
    p->lane();
    checked(cudaMemcpy(p->valid.get(),prefixes.data(),s.batch*sizeof(int),cudaMemcpyHostToDevice),"Mamba3 projected prefix upload");
    record_gpu_transfer(Device::GPU,Device::CPU,s.batch*sizeof(int));
    auto& q=p->prepared;
    if (!pg::forward(s,c,p->source(),{q.q.raw_data(),q.k.raw_data(),q.adt.raw_data(),q.dt.raw_data(),q.angles.raw_data(),p->inverse.raw_data()},p->status.get()))
        throw std::runtime_error("Mamba3 preprocessing enqueue failed; discard and drain lane");
    p->recurrence=mamba3_siso::Tape::forward(s,q,initial,prefixes,p->status.get());
    p->copy_status();gpu::record_dispatch(gpu::DispatchPath::Mamba3PreprocessForward);
    return std::unique_ptr<Tape>(new Tape(std::move(p)));
#else
    throw std::runtime_error("Mamba3 projected tape requires a GPU build; no CPU fallback");
#endif
}
Tensor Tape::output() const {impl_->lane();return impl_->recurrence->output();}
State Tape::snapshot_final_state() const {impl_->lane();return impl_->recurrence->snapshot_final_state();}
Gradients Tape::backward(const Tensor& dy,const State& seed) {
    impl_->lane();if (impl_->consumed) throw std::runtime_error("Mamba3 projected tape consumed/cancelled");
    shape(dy,impl_->input.v.shape.dims,"output adjoint");validate_state(impl_->shape,seed);
#ifdef USE_CUDA
    impl_->consumed=true; // any partial enqueue must be single-use even on error
    auto rec=impl_->recurrence->backward(dy,seed);impl_->copy_status();
    auto make=[](const Tensor& t) {return Tensor::uninitialized(t.shape.dims,Device::GPU);};
    auto& x=impl_->input;
    impl_->gradients={{make(x.q),make(x.k),rec.input.v,rec.input.z,make(x.raw_a),make(x.raw_dt),rec.input.trap,make(x.angles),
        make(x.q_norm),make(x.k_norm),make(x.dt_bias),rec.input.q_bias,rec.input.k_bias,rec.input.d},rec.initial_state};
    auto& result=impl_->gradients;
    pg::Adjoint adj{rec.input.q.raw_data(),rec.input.k.raw_data(),rec.input.adt.raw_data(),rec.input.dt.raw_data(),rec.input.angles.raw_data()};
    const auto raw=destination(result.input);
    if (!pg::backward(impl_->shape,impl_->config,impl_->source(),adj,raw,impl_->partials.raw_data(),impl_->status.get()))
        throw std::runtime_error("Mamba3 preprocessing VJP enqueue failed; discard and drain lane");
    mamba3_siso_gpu::Gradient r{nullptr,nullptr,result.input.v.raw_data(),result.input.z.size?result.input.z.raw_data():nullptr,
        nullptr,nullptr,result.input.trap.raw_data(),nullptr,result.input.q_bias.raw_data(),result.input.k_bias.raw_data(),result.input.d.size?result.input.d.raw_data():nullptr};
    auto& di=result.initial_state;
    if (!pg::gate_chain(impl_->shape,impl_->status.get(),raw,r,{di.phase.raw_data(),di.ssm.raw_data(),di.k.raw_data(),di.v.raw_data()}))
        throw std::runtime_error("Mamba3 composed gradient gate enqueue failed; discard and drain lane");
    gpu::record_dispatch(gpu::DispatchPath::Mamba3PreprocessBackward);return result;
#else
    throw std::runtime_error("Mamba3 projected backward requires GPU build");
#endif
}
void Tape::cancel() {impl_->lane();impl_->consumed=true;impl_->recurrence->cancel();}
bool Tape::consumed() const {return impl_->consumed;}
const int* Tape::device_status() const {
    impl_->lane();
#ifdef USE_CUDA
    return impl_->status.get();
#else
    return nullptr;
#endif
}
std::vector<int> Tape::audit_status() const {
    impl_->lane();
#ifdef USE_CUDA
    std::vector<int> out(impl_->shape.batch);
    checked(cudaMemcpyAsync(out.data(),impl_->status.get(),out.size()*sizeof(int),cudaMemcpyDeviceToHost,impl_->stream),"Mamba3 projected explicit audit");
    record_gpu_transfer(Device::CPU,Device::GPU,out.size()*sizeof(int));
    checked(cudaStreamSynchronize(impl_->stream),"Mamba3 projected explicit audit fence");record_gpu_stream_synchronization();return out;
#else
    return {};
#endif
}
std::size_t Tape::preprocessing_workspace_bytes() const {return (impl_->inverse.size+impl_->partials.size)*sizeof(float);}
}
