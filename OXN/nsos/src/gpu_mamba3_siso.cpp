#include "gpu_mamba3_siso.h"
#include "gpu_execution.h"
#include "cuda/device_buffer.h"
#include <climits>
#include <thread>
#include <stdexcept>
#include <utility>

namespace nsos::mamba3_siso {
namespace {
namespace mg=mamba3_siso_gpu;
bool empty(const State& s) {return s.phase.size==0&&s.ssm.size==0&&s.k.size==0&&s.v.size==0;}
void shape(const Tensor& x,const std::vector<int>& dims,const char* name) {
    if (x.get_device()!=Device::GPU||x.shape.dims!=dims)
        throw std::invalid_argument(std::string("Mamba3 SISO GPU tensor shape/device: ")+name);
    std::size_t count=1;
    for (int i=static_cast<int>(dims.size())-1;i>=0;--i) {
        if (x.shape.strides.size()!=dims.size()||x.shape.strides[i]!=count||dims[i]<=0||
            count>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(dims[i]))
            throw std::invalid_argument(std::string("Mamba3 SISO noncontiguous/overflow tensor: ")+name);
        count*=dims[i];
    }
    if (x.size!=count||!x.raw_data()) throw std::invalid_argument(std::string("Mamba3 SISO tensor extent/storage: ")+name);
}
void validate_operands(const Shape& s,const Operands& x) {
    shape(x.q,{s.batch,s.sequence,s.groups,s.state_dim},"Q");shape(x.k,x.q.shape.dims,"K");
    shape(x.v,{s.batch,s.sequence,s.heads,s.head_dim},"V");if (x.z.size) shape(x.z,x.v.shape.dims,"Z");
    shape(x.adt,{s.batch,s.sequence,s.heads},"ADT");shape(x.dt,x.adt.shape.dims,"DT");shape(x.trap,x.adt.shape.dims,"Trap");
    shape(x.angles,{s.batch,s.sequence,s.heads,s.rotary_pairs},"Angles");
    shape(x.q_bias,{s.heads,s.state_dim},"Q Bias");shape(x.k_bias,x.q_bias.shape.dims,"K Bias");if (x.d.size) shape(x.d,{s.heads},"D");
}
void validate_state(const Shape& s,const State& x) {
    if (empty(x)) return;
    shape(x.phase,{s.batch,s.heads,s.rotary_pairs},"Phase state");shape(x.ssm,{s.batch,s.heads,s.head_dim,s.state_dim},"SSM state");
    shape(x.k,{s.batch,s.heads,s.state_dim},"K state");shape(x.v,{s.batch,s.heads,s.head_dim},"V state");
}
State clone(const State& s) {if (empty(s)) return {};return {s.phase.clone(),s.ssm.clone(),s.k.clone(),s.v.clone()};}
Operands clone(const Operands& x) {
    return {x.q.clone(),x.k.clone(),x.v.clone(),x.z.size?x.z.clone():Tensor(),x.adt.clone(),x.dt.clone(),x.trap.clone(),
        x.angles.clone(),x.q_bias.clone(),x.k_bias.clone(),x.d.size?x.d.clone():Tensor()};
}
State allocate_state(const Shape& s) {
    return {Tensor::uninitialized({s.batch,s.heads,s.rotary_pairs},Device::GPU),Tensor::uninitialized({s.batch,s.heads,s.head_dim,s.state_dim},Device::GPU),
        Tensor::uninitialized({s.batch,s.heads,s.state_dim},Device::GPU),Tensor::uninitialized({s.batch,s.heads,s.head_dim},Device::GPU)};
}
mg::ConstState pointers(const State& s) {if (empty(s)) return {};return {s.phase.raw_data(),s.ssm.raw_data(),s.k.raw_data(),s.v.raw_data()};}
mg::State destination(State& s) {return {s.phase.raw_data(),s.ssm.raw_data(),s.k.raw_data(),s.v.raw_data()};}
mg::Gradient destination(Operands& s) {
    return {s.q.raw_data(),s.k.raw_data(),s.v.raw_data(),s.z.size?s.z.raw_data():nullptr,s.adt.raw_data(),s.dt.raw_data(),s.trap.raw_data(),
        s.angles.raw_data(),s.q_bias.raw_data(),s.k_bias.raw_data(),s.d.size?s.d.raw_data():nullptr};
}
Operands allocate_like(const Operands& x) {
    auto make=[](const Tensor& t) {return t.size?Tensor::uninitialized(t.shape.dims,Device::GPU):Tensor();};
    return {make(x.q),make(x.k),make(x.v),make(x.z),make(x.adt),make(x.dt),make(x.trap),make(x.angles),make(x.q_bias),make(x.k_bias),make(x.d)};
}
#ifdef USE_CUDA
void check(cudaError_t error,const char* name) {if (error!=cudaSuccess) throw std::runtime_error(std::string(name)+": "+cudaGetErrorString(error));}
#endif
}
struct Tape::Impl {
    Shape shape;Operands input;State initial,final,seed;Gradients gradients;Tensor output,dy,boundaries,replay,partials;
    bool consumed=false;std::thread::id owner=std::this_thread::get_id();
#ifdef USE_CUDA
    int device=-1;cudaStream_t stream=nullptr;
    cuda_detail::DeviceBuffer<int> valid,status,upstream;
    void lane() const {
        int current=-1;check(cudaGetDevice(&current),"Mamba3 tape current device");
        if (std::this_thread::get_id()!=owner||current!=device||gpu::current_stream()!=stream)
            throw std::runtime_error("Mamba3 tape requires its original thread/device/stream lane");
    }
    mg::Input operands() const {
        return {input.q.raw_data(),input.k.raw_data(),input.v.raw_data(),input.z.size?input.z.raw_data():nullptr,
            input.adt.raw_data(),input.dt.raw_data(),input.trap.raw_data(),input.angles.raw_data(),input.q_bias.raw_data(),input.k_bias.raw_data(),
            input.d.size?input.d.raw_data():nullptr,valid.get()};
    }
#else
    void lane() const {throw std::runtime_error("Mamba3 SISO tape requires a GPU build");}
#endif
};
Tape::Tape(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
Tape::~Tape()=default;
std::unique_ptr<Tape> Tape::forward(const Shape& s,const Operands& x,const State& initial,const std::vector<int>& lengths,const int* upstream) {
    if (!mg::eligible(s)) throw std::invalid_argument("Mamba3 SISO unsupported geometry/chunk");
    validate_operands(s,x);validate_state(s,initial);
    auto valid=lengths.empty()?std::vector<int>(s.batch,s.sequence):lengths;
    if (valid.size()!=static_cast<std::size_t>(s.batch)) throw std::invalid_argument("Mamba3 SISO prefix shape");
    for (int n:valid) if (n<0||n>s.sequence) throw std::invalid_argument("Mamba3 SISO prefix out of range");
    for (auto count:{mg::boundary_elements(s),mg::replay_elements(s),mg::gradient_partial_elements(s)})
        if (count>INT_MAX) throw std::length_error("Mamba3 SISO flat workspace exceeds Tensor dimension ABI");
#ifdef USE_CUDA
    if (!mg::supported(s)) throw std::runtime_error("Mamba3 SISO requires compiled RDNA3 wave32 device capability");
    auto p=std::make_unique<Impl>();p->shape=s;
    check(cudaGetDevice(&p->device),"Mamba3 tape select device");p->stream=gpu::current_stream();
    p->input=clone(x);p->initial=clone(initial);p->final=allocate_state(s);
    p->output=Tensor::uninitialized(x.v.shape.dims,Device::GPU);
    p->boundaries=Tensor::uninitialized({static_cast<int>(mg::boundary_elements(s))},Device::GPU);
    p->replay=Tensor::uninitialized({static_cast<int>(mg::replay_elements(s))},Device::GPU);
    p->partials=Tensor::uninitialized({static_cast<int>(mg::gradient_partial_elements(s))},Device::GPU);
    if (!p->valid.ensure(s.batch)||!p->status.ensure(s.batch)) throw std::runtime_error("Mamba3 tape metadata allocation failed");
    p->lane();
    // Synchronous metadata H2D owns its tiny host source lifetime. Do not hide
    // an async pageable host vector whose storage vanishes on return.
    check(cudaMemcpy(p->valid.get(),valid.data(),valid.size()*sizeof(int),cudaMemcpyHostToDevice),"Mamba3 prefix upload");
    record_gpu_transfer(Device::GPU,Device::CPU,valid.size()*sizeof(int));
    if (upstream) {
        if (!p->upstream.ensure(s.batch)) throw std::runtime_error("Mamba3 upstream status allocation failed");
        check(cudaMemcpyAsync(p->upstream.get(),upstream,s.batch*sizeof(int),cudaMemcpyDeviceToDevice,p->stream),"Mamba3 upstream status clone");
        record_gpu_transfer(Device::GPU,Device::GPU,s.batch*sizeof(int));
    }
    if (!mg::forward(s,p->operands(),pointers(p->initial),destination(p->final),p->output.raw_data(),p->boundaries.raw_data(),p->status.get(),p->upstream.get()))
        throw std::runtime_error("Mamba3 SISO forward enqueue failed; discard tape and drain lane");
    gpu::record_dispatch(gpu::DispatchPath::Mamba3SisoForward);
    return std::unique_ptr<Tape>(new Tape(std::move(p)));
#else
    throw std::runtime_error("Mamba3 SISO requires a GPU build; no CPU fallback");
#endif
}
Tensor Tape::output() const {impl_->lane();return impl_->output;}
State Tape::snapshot_final_state() const {impl_->lane();return clone(impl_->final);}
Gradients Tape::backward(const Tensor& dy,const State& seed) {
    impl_->lane();if (impl_->consumed) throw std::runtime_error("Mamba3 SISO tape already consumed/cancelled");
    shape(dy,impl_->input.v.shape.dims,"output adjoint");validate_state(impl_->shape,seed);
#ifdef USE_CUDA
    impl_->dy=dy.clone();impl_->seed=clone(seed);
    impl_->gradients={allocate_like(impl_->input),allocate_state(impl_->shape)};
    auto& result=impl_->gradients;
    impl_->consumed=true; // launch failure can queue partial work; never reuse
    if (!mg::backward(impl_->shape,impl_->operands(),pointers(impl_->initial),impl_->dy.raw_data(),pointers(impl_->seed),
        destination(result.input),destination(result.initial_state),impl_->boundaries.raw_data(),impl_->replay.raw_data(),impl_->partials.raw_data(),impl_->status.get()))
        throw std::runtime_error("Mamba3 SISO backward enqueue failed; discard result and drain lane");
    gpu::record_dispatch(gpu::DispatchPath::Mamba3SisoBackward);return result;
#else
    throw std::runtime_error("Mamba3 SISO backward requires a GPU build");
#endif
}
void Tape::cancel() {impl_->lane();impl_->consumed=true;}
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
    std::vector<int> result(impl_->shape.batch);
    check(cudaMemcpyAsync(result.data(),impl_->status.get(),result.size()*sizeof(int),cudaMemcpyDeviceToHost,impl_->stream),"Mamba3 explicit status audit");
    record_gpu_transfer(Device::CPU,Device::GPU,result.size()*sizeof(int));
    check(cudaStreamSynchronize(impl_->stream),"Mamba3 explicit audit fence");record_gpu_stream_synchronization();return result;
#else
    return {};
#endif
}
std::size_t Tape::boundary_bytes() const {return mg::boundary_elements(impl_->shape)*sizeof(float);}
std::size_t Tape::replay_bytes() const {return mg::replay_elements(impl_->shape)*sizeof(float);}
std::size_t Tape::partial_bytes() const {return mg::gradient_partial_elements(impl_->shape)*sizeof(float);}
}
