#include "mamba3_layer.h"
#include "mamba3_layer_math.h"
#include "cuda/mamba3_projection_wmma.cuh"
#include "gpu_backend.h"
#include "gpu_execution.h"
#include "cuda/device_buffer.h"
#include "checkpoint_io.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <thread>

namespace nsos {
namespace {
namespace mb=mamba3_block;
namespace mp=mamba3_projection;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void tensor_shape(const Tensor& x,const std::vector<int>& dims,Device dev) {
    require(x.shape.dims==dims&&x.get_device()==dev&&x.raw_data(),"Mamba3 tensor shape/device/storage mismatch");
    std::size_t n=1;for(int i=int(dims.size())-1;i>=0;--i) {require(x.shape.strides[i]==n,"Mamba3 requires contiguous tensors");n*=dims[i];}
    require(x.size==n,"Mamba3 tensor extent mismatch");
}
bool empty(const Mamba3State& s) {return !(s.phase.size||s.ssm.size||s.k.size||s.v.size);}
Mamba3State clone(const Mamba3State& s) {if(empty(s)) return {};return {s.phase.clone(),s.ssm.clone(),s.k.clone(),s.v.clone()};}
Mamba3State move(const Mamba3State& s,Device dev) {if(empty(s)) return {};return {s.phase.to(dev),s.ssm.to(dev),s.k.to(dev),s.v.to(dev)};}
Mamba3State zero(const mb::Shape& s,Device dev) {return {Tensor::zeros({s.batch,s.heads,s.rotary_pairs},dev),Tensor::zeros({s.batch,s.heads,s.head_dim,s.state_dim},dev),Tensor::zeros({s.batch,s.heads,s.rank,s.state_dim},dev),Tensor::zeros({s.batch,s.heads,s.head_dim},dev)};}
void state_shape(const Mamba3State& x,const mb::Shape& s,Device dev) {
    if(empty(x)) return;
    tensor_shape(x.phase,{s.batch,s.heads,s.rotary_pairs},dev);tensor_shape(x.ssm,{s.batch,s.heads,s.head_dim,s.state_dim},dev);
    tensor_shape(x.k,{s.batch,s.heads,s.rank,s.state_dim},dev);tensor_shape(x.v,{s.batch,s.heads,s.head_dim},dev);
}
mb::State<const float> source(const Mamba3State& s) {return {s.phase.raw_data(),s.ssm.raw_data(),s.k.raw_data(),s.v.raw_data()};}
mb::State<float> dest(Mamba3State& s) {return {s.phase.raw_data(),s.ssm.raw_data(),s.k.raw_data(),s.v.raw_data()};}
Tensor flat(std::size_t n,Device dev) {require(n>0&&n<=INT_MAX,"Mamba3 scratch exceeds Tensor capacity");return Tensor::zeros({int(n)},dev);}
mb::Shape geometry(int model,const Mamba3Config& c,int B=1,int S=1) {
    require(c.schema_version==1&&c.expand>0&&c.expand<=16&&model>0&&model<=65536&&c.head_dim>0,"Mamba3 configuration schema/expansion");
    const int inner=model*c.expand;require(inner%c.head_dim==0,"Mamba3 inner dimension/head dimension");
    require(c.rope_fraction==0.5f||c.rope_fraction==1.0f,"Mamba3 RoPE fraction must be 0.5 or 1");
    const int pairs=int(c.state_dim*c.rope_fraction)/2;
    mb::Shape s{B,S,model,inner/c.head_dim,c.n_groups,c.head_dim,c.state_dim,c.mimo?c.mimo_rank:1,pairs,c.mimo,c.outproj_norm,c.norm_eps,c.a_floor};
    require(mb::eligible(s),"Mamba3 unsupported/overflowing configuration");
    require(std::isfinite(c.dt_min)&&std::isfinite(c.dt_max)&&std::isfinite(c.dt_init_floor)&&c.dt_min>0&&c.dt_max>=c.dt_min&&c.dt_max<=16&&c.dt_init_floor>0&&c.dt_init_floor<=c.dt_max,"Mamba3 DT initialization range");return s;
}
std::string identity(int model,const Mamba3Config& c) {
    auto s=geometry(model,c);std::ostringstream o;o<<mb::identity<<":"<<mb::upstream<<":"<<c.schema_version<<":"<<model<<":"<<c.expand<<":"<<s.heads<<":"<<c.head_dim<<":"<<c.state_dim<<":"<<c.n_groups<<":"<<s.rank<<":"<<c.mimo<<":"<<c.outproj_norm<<":"<<std::hexfloat<<c.rope_fraction<<":"<<c.norm_eps<<":"<<c.a_floor;return o.str();
}
void mask_cpu(const Tensor& src,Tensor& dst,int B,int S,int W,const std::vector<int>& valid) {
    const float* p=src.data();float* q=dst.data();for(int b=0;b<B;++b) for(int t=0;t<S;++t) for(int i=0;i<W;++i) {const auto pos=(std::size_t(b)*S+t)*W+i;q[pos]=t<valid[b]?p[pos]:0;}
}
void check_cpu(const Tensor& src,Tensor& status,int B,int S,int W,const std::vector<int>& valid) {
    const float* x=src.data();float* code=status.data();for(int b=0;b<B;++b) for(std::size_t i=0;i<std::size_t(valid[b])*W;++i) if(!std::isfinite(x[std::size_t(b)*S*W+i])) code[b]=3;
}
void gate_cpu(Tensor& values,const Tensor& status,int B,bool whole=false) {
    bool bad=false;for(int b=0;b<B;++b) bad|=status.data()[b]!=0;
    auto p=values.data();for(int b=0;b<B;++b) if((whole&&bad)||status.data()[b]) std::fill_n(p+std::size_t(b)*(values.size/B),values.size/B,0.f);
}
#ifdef USE_CUDA
void checked(cudaError_t e,const char* message) {if(e!=cudaSuccess) throw std::runtime_error(std::string(message)+": "+cudaGetErrorString(e));}
#endif
struct CoreField {std::string name;std::size_t offset;std::vector<int> shape;};
std::vector<CoreField> fields(const mb::Shape& s) {
    const mb::Layout l(s);std::vector<CoreField> f{{"B_norm.weight",l.bnorm,{s.state_dim}},{"C_norm.weight",l.cnorm,{s.state_dim}},{"dt_bias",l.dt,{s.heads}},
        {"B_bias",l.bbias,{s.heads,s.rank,s.state_dim}},{"C_bias",l.cbias,{s.heads,s.rank,s.state_dim}},{"D",l.d,{s.heads}}};
    if(s.mimo) {f.push_back({"mimo_x",l.x,{s.heads,s.rank,s.head_dim}});f.push_back({"mimo_z",l.z,{s.heads,s.rank,s.head_dim}});f.push_back({"mimo_o",l.o,{s.heads,s.rank,s.head_dim}});}
    if(s.out_norm) f.push_back({"norm.weight",l.norm,{s.inner()}});return f;
}
}
struct Mamba3Tape::Impl {
    mb::Shape shape;Device device=Device::CPU;
    mp::Policy projection_policy=mp::Policy::ExactFP32;std::uint64_t projection_dispatches=0;std::uint64_t model_id=0;bool consumed=false,published=false,cancelled=false;
    std::vector<std::uint64_t> versions;std::vector<CoreField> core_fields;std::vector<int> valid;std::vector<int> input_shape;
    Tensor input,in_weight,out_weight,core,projection,mixed,output,status,history,q,k,phase,readout;
    Tensor adjoint,dmixed,dprojection,partial,dcore,scratch,din_weight,dout_weight,dinput;
    mb::GpuProvider gpu_provider=mb::GpuProvider::DenseReference;
    Tensor gy,token_parameters,bc_gradient,phase_gradient,reverse,dz,dx_value,ddt,da,dtrap,coefficients;
    // Scratch belongs to the consumed tape; backward reuses it, not primal data.
    mutable Tensor hierarchy_a,hierarchy_b;
    mb::BackwardWorkspace backward_workspace() {return {gy.raw_data(),token_parameters.raw_data(),bc_gradient.raw_data(),phase_gradient.raw_data(),reverse.raw_data(),dz.raw_data(),dx_value.raw_data(),ddt.raw_data(),da.raw_data(),dtrap.raw_data()};}
    Mamba3State initial,final,dinitial,seed;
    std::thread::id thread=std::this_thread::get_id();
#ifdef USE_CUDA
    int device_id=-1;cudaStream_t stream=nullptr;cuda_detail::DeviceBuffer<int> prefixes;
#endif
    void lane() const {
        require(thread==std::this_thread::get_id(),"Mamba3 tape requires original thread");
#ifdef USE_CUDA
        if(device==Device::GPU) {int id=-1;checked(cudaGetDevice(&id),"Mamba3 selected device");require(id==device_id&&gpu::current_stream()==stream,"Mamba3 tape requires original device/stream");}
#endif
    }
    void mask(const Tensor& src,Tensor& dst,int width) {
        if(device==Device::CPU) mask_cpu(src,dst,shape.batch,shape.sequence,width,valid);
#ifdef USE_CUDA
        else require(mb::gpu_mask(shape.batch,shape.sequence,width,prefixes.get(),src.raw_data(),dst.raw_data()),"Mamba3 mask enqueue failed; discard/drain lane");
#endif
    }
    void check(const Tensor& src,int width) {
        if(device==Device::CPU) check_cpu(src,status,shape.batch,shape.sequence,width,valid);
#ifdef USE_CUDA
        else require(mb::gpu_check(shape.batch,shape.sequence,width,prefixes.get(),src.raw_data(),status.raw_data()),"Mamba3 finite check enqueue failed; discard/drain lane");
#endif
    }
    void gate(Tensor& values,bool whole=false) {
        if(device==Device::CPU) gate_cpu(values,status,shape.batch,whole);
#ifdef USE_CUDA
        else require(mb::gpu_gate(shape.batch,std::size_t(values.size)/shape.batch,status.raw_data(),values.raw_data(),whole),"Mamba3 gate enqueue failed; discard/drain lane");
#endif
    }
    void gate_state(Mamba3State& state) {gate(state.phase);gate(state.ssm);gate(state.k);gate(state.v);}
    Tensor dense(const Tensor& a,const Tensor& b,bool ta,bool tb,mp::ExactAxis axis=mp::ExactAxis::None,int begin=0) {
        const auto& ad=a.shape.dims;const auto& bd=b.shape.dims;
        require(ad.size()==2&&bd.size()==2,"Mamba3 dense projection requires matrices");
        const int rows=ad[ta?1:0],k=ad[ta?0:1],cols=bd[tb?0:1];
        require(k==bd[tb?1:0],"Mamba3 dense projection reduction mismatch");
        if(projection_policy==mp::Policy::ExactFP32) {
            if(ta) {require(!tb,"Mamba3 exact transpose-both unsupported");return matmul_tn(a,b);}
            return tb?matmul_nt(a,b):a.matmul(b);
        }
        require(device==Device::GPU,"Mamba3 lowp projection requires GPU; no host fallback");
        // The WMMA pass writes every output cell, including zero-padded M/N/K
        // tiles; the exact reduction suffix reads only this freshly written pass.
        Tensor out=Tensor::uninitialized({rows,cols},device);
#ifdef USE_CUDA
        require(mp::gemm(projection_policy,ta,tb,rows,cols,k,a.raw_data(),b.raw_data(),out.raw_data(),axis,begin),
            "Mamba3 lowp projection unavailable/enqueue failed; no exact fallback; discard/drain lane");
        ++projection_dispatches;
#else
        throw std::runtime_error("Mamba3 lowp projection requires GPU build");
#endif
        return out;
    }
    mb::Trace<float> trace() {return {history.raw_data(),q.raw_data(),k.raw_data(),phase.raw_data(),readout.raw_data(),gpu_provider!=mb::GpuProvider::DenseReference,mb::is_flash_provider(gpu_provider),coefficients.raw_data(),mb::is_replay_lds_provider(gpu_provider),mb::is_hierarchical_provider(gpu_provider),hierarchy_a.raw_data(),hierarchy_b.raw_data()};}
    mb::Trace<const float> trace_const() const {return {history.raw_data(),q.raw_data(),k.raw_data(),phase.raw_data(),readout.raw_data(),gpu_provider!=mb::GpuProvider::DenseReference,mb::is_flash_provider(gpu_provider),coefficients.raw_data(),mb::is_replay_lds_provider(gpu_provider),mb::is_hierarchical_provider(gpu_provider),hierarchy_a.raw_data(),hierarchy_b.raw_data()};}
    std::size_t workspace() const {return (std::size_t(history.size)+coefficients.size+q.size+k.size+phase.size+readout.size+partial.size+scratch.size+gy.size+token_parameters.size+bc_gradient.size+phase_gradient.size+reverse.size+dz.size+dx_value.size+ddt.size+da.size+dtrap.size+hierarchy_a.size+hierarchy_b.size)*sizeof(float);}
};
struct Mamba3Layer::Impl {
    int model;Mamba3Config config;Device device=Device::CPU;Tensor core;std::vector<Parameter> params;std::string name="mamba3";
    bool training=true,streaming=false;Mamba3State session;std::shared_ptr<Mamba3Tape> pending;
    Mamba3Telemetry telemetry;std::uint64_t id;
    void rebind() {const auto f=fields(geometry(model,config));for(std::size_t i=0;i<f.size();++i) params[i+2].data=core.storage_view(f[i].offset,f[i].shape);}
};
Mamba3Tape::Mamba3Tape(std::unique_ptr<Impl> p):impl_(std::move(p)) {}
Mamba3Tape::~Mamba3Tape()=default;
Tensor Mamba3Tape::output() const {impl_->lane();return impl_->output.clone().reshape(impl_->input_shape);}
Mamba3State Mamba3Tape::snapshot_final_state() const {impl_->lane();return clone(impl_->final);}
const Tensor& Mamba3Tape::status_tensor() const {impl_->lane();return impl_->status;}
bool Mamba3Tape::consumed() const {return impl_->consumed;}
void Mamba3Tape::cancel() {impl_->lane();impl_->consumed=true;impl_->cancelled=true;}
std::size_t Mamba3Tape::workspace_bytes() const {return impl_->workspace();}
std::string Mamba3Tape::projection_runtime_identity() const {return mp::identity(impl_->projection_policy);}
std::vector<int> Mamba3Tape::audit_status() const {
    impl_->lane();Tensor status=impl_->status.cpu();std::vector<int> codes(impl_->shape.batch);
    for(int b=0;b<impl_->shape.batch;++b) codes[b]=int(status.data()[b]);return codes;
}
Mamba3Backward Mamba3Tape::backward(const Tensor& dy,const Mamba3State& seed) {
    auto& p=*impl_;p.lane();require(!p.consumed,"Mamba3 tape already consumed/cancelled");
    tensor_shape(dy,p.input_shape,p.device);state_shape(seed,p.shape,p.device);require(matmul_precision_mode()==0,"Mamba3 v1 requires FP32 GEMM policy");
    p.consumed=true;const auto& s=p.shape;const mb::Layout l(s);const int rows=s.batch*s.sequence;
    p.seed=empty(seed)?zero(s,p.device):clone(seed);p.adjoint=Tensor::zeros({rows,s.model},p.device);p.mask(dy,p.adjoint,s.model);p.check(p.adjoint,s.model);
    p.dmixed=p.dense(p.adjoint,p.out_weight,false,false); // [T,D] * [D,I]
    p.dprojection=Tensor::zeros({rows,s.width()},p.device);p.partial=flat(std::size_t(s.batch)*l.total,p.device);p.dcore=flat(l.total,p.device);
    p.scratch=flat(std::size_t(s.batch)*mb::scratch_per_head(s),p.device);p.dinitial=zero(s,p.device);
    if(p.gpu_provider!=mb::GpuProvider::DenseReference) {
        const auto tokens=std::size_t(s.batch)*s.heads*s.sequence;
        p.gy=flat(mb::readout_size(s),p.device);p.token_parameters=flat(tokens*mb::token_layout(s).total,p.device);
        p.bc_gradient=flat(tokens*2*s.rank*s.state_dim,p.device);p.phase_gradient=flat(mb::phase_size(s),p.device);
        p.reverse=flat(mb::is_flash_provider(p.gpu_provider)?mb::checkpoint_history_size(s):mb::history_size(s),p.device);
        p.dz=flat(tokens*s.head_dim,p.device);p.dx_value=flat(tokens*s.head_dim,p.device);
        p.ddt=flat(tokens,p.device);p.da=flat(tokens,p.device);p.dtrap=flat(tokens,p.device);
    }
    if(p.device==Device::CPU) {
        for(int b=0;b<s.batch;++b) mb::detail::backward_batch(s,l,b,p.projection.data(),p.core.data(),p.valid.data(),source(p.initial),p.trace_const(),p.dmixed.data(),source(p.seed),p.dprojection.data(),p.partial.data()+std::size_t(b)*l.total,dest(p.dinitial),p.scratch.data()+std::size_t(b)*mb::scratch_per_head(s),p.status.data());
        bool bad=false;for(int b=0;b<s.batch;++b) bad|=p.status.data()[b]!=0;
        if(!bad) for(int b=0;b<s.batch;++b) for(std::size_t i=0;i<l.total;++i) p.dcore.data()[i]+=p.partial.data()[std::size_t(b)*l.total+i];
    }
#ifdef USE_CUDA
    else {require(mb::gpu_backward(s,p.projection.raw_data(),p.core.raw_data(),p.prefixes.get(),source(p.initial),p.trace_const(),p.dmixed.raw_data(),source(p.seed),p.dprojection.raw_data(),p.partial.raw_data(),dest(p.dinitial),p.scratch.raw_data(),p.status.raw_data(),p.backward_workspace()),"Mamba3 VJP enqueue failed; discard/drain lane");require(mb::gpu_reduce(s,p.partial.raw_data(),p.dcore.raw_data(),p.status.raw_data()),"Mamba3 reduction enqueue failed");}
#endif
    p.gate(p.dprojection);p.gate_state(p.dinitial);
    p.dinput=p.dense(p.dprojection,p.in_weight,false,false,mp::ExactAxis::Reduction,2*s.inner()+2*s.bc());p.check(p.dinput,s.model);p.gate(p.dinput);p.gate_state(p.dinitial);
    p.din_weight=p.dense(p.dprojection,p.input,true,false,mp::ExactAxis::OutputRows,2*s.inner()+2*s.bc());p.dout_weight=p.dense(p.adjoint,p.mixed,true,false);
    // Projection GEMM/parameter reductions can overflow after the recurrence.
    // Check every parameter gradient and propagate failure to ALL batches.
    auto finite_parameters=[&](Tensor& t) {
        if(p.device==Device::CPU) {bool bad=false;for(std::size_t i=0;i<std::size_t(t.size);++i) bad|=!std::isfinite(t.data()[i]);if(bad) for(int b=0;b<s.batch;++b) p.status.data()[b]=3;}
#ifdef USE_CUDA
        else require(mb::gpu_parameter_check(s.batch,std::size_t(t.size),t.raw_data(),p.status.raw_data()),"Mamba3 parameter finite check enqueue failed");
#endif
    };
    finite_parameters(p.din_weight);finite_parameters(p.dout_weight);finite_parameters(p.dcore);
    auto whole_parameter_gate=[&](Tensor& t) {
        if(p.device==Device::CPU) {bool bad=false;for(int b=0;b<s.batch;++b) bad|=p.status.data()[b]!=0;if(bad) std::fill_n(t.data(),t.size,0.f);}
#ifdef USE_CUDA
        else require(mb::gpu_parameter_gate(s.batch,std::size_t(t.size),p.status.raw_data(),t.raw_data()),"Mamba3 parameter gate enqueue failed");
#endif
    };
    whole_parameter_gate(p.din_weight);whole_parameter_gate(p.dout_weight);whole_parameter_gate(p.dcore);p.gate(p.dinput);p.gate_state(p.dinitial);
    Mamba3Backward result{p.dinput.clone().reshape(p.input_shape),clone(p.dinitial),{p.din_weight.clone(),p.dout_weight.clone()},shared_from_this()};
    for(const auto& f:p.core_fields) result.parameters.push_back(p.dcore.storage_view(f.offset,f.shape).clone());return result;
}
Mamba3Layer::Mamba3Layer(int model,const Mamba3Config& config):impl_(std::make_unique<Impl>()) {
    const auto s=geometry(model,config);const mb::Layout l(s);auto& p=*impl_;p.model=model;p.config=config;
    static std::atomic<std::uint64_t> ids{1};p.id=ids.fetch_add(1);p.core=Tensor::ones({int(l.total)},Device::CPU);
    p.params.emplace_back(Tensor::kaiming_uniform({s.width(),model},Device::CPU,config.seed),"in_proj.weight");
    p.params.emplace_back(Tensor::kaiming_uniform({model,s.inner()},Device::CPU,config.seed+1),"out_proj.weight");
    std::mt19937_64 rng(config.seed+2);std::uniform_real_distribution<double> uniform(0,1);
    for(int h=0;h<s.heads;++h) {const double dt=std::max(double(config.dt_init_floor),std::exp(std::log(config.dt_min)+uniform(rng)*(std::log(config.dt_max)-std::log(config.dt_min))));p.core.data()[l.dt+h]=float(dt+std::log(-std::expm1(-dt)));}
    if(s.mimo) for(int i=0;i<s.heads*s.rank*s.head_dim;++i) {p.core.data()[l.x+i]=1.f/s.rank;p.core.data()[l.o+i]=1.f/s.rank;}
    for(const auto& f:fields(s)) p.params.emplace_back(p.core.storage_view(f.offset,f.shape),f.name);
    for(auto& param:p.params) param.track_gradient_contributions();
}
Mamba3Layer::~Mamba3Layer()=default;
std::vector<Parameter*> Mamba3Layer::parameters() {std::vector<Parameter*> p;for(auto& item:impl_->params) p.push_back(&item);return p;}
std::vector<Parameter*> Mamba3Layer::no_weight_decay_parameters() {return {&impl_->params[4],&impl_->params[7]};}
std::shared_ptr<Mamba3Tape> Mamba3Layer::forward_owned(const Tensor& input,const Mamba3State& initial,const std::vector<int>& lengths) {
    auto& layer=*impl_;require(matmul_precision_mode()==0,"Mamba3 v1 requires FP32 GEMM policy");
    const auto dims=input.shape.dims;require(dims.size()==2||dims.size()==3,"Mamba3 input must be [S,D] or [B,S,D]");
    const int B=dims.size()==3?dims[0]:1,S=dims.size()==3?dims[1]:dims[0];const auto s=geometry(layer.model,layer.config,B,S);
    tensor_shape(input,dims,layer.device);require(dims.back()==layer.model,"Mamba3 model dimension mismatch");state_shape(initial,s,layer.device);
    auto p=std::make_unique<Mamba3Tape::Impl>();p->shape=s;p->device=layer.device;
    p->gpu_provider=mb::gpu_provider_from_environment();
    require(p->gpu_provider==mb::GpuProvider::DenseReference||layer.device==Device::GPU,"Mamba3 optimized provider requires GPU; no CPU fallback");
    require(p->gpu_provider==mb::GpuProvider::DenseReference||mb::parallel_eligible(s),"Mamba3 optimized provider workspace/grid capacity exceeded");require(!mb::is_hierarchical_provider(p->gpu_provider)||mb::hierarchical_eligible(s),"Mamba3 hierarchical workspace capacity exceeded");p->model_id=layer.id;p->input_shape=dims;p->core_fields=fields(s);
    p->projection_policy=mp::policy();
    if(p->projection_policy!=mp::Policy::ExactFP32) {
        require(layer.device==Device::GPU,"Mamba3 lowp projection requires GPU; no host fallback");
        for(const auto& projection_dims:{std::array<int,3>{B*S,s.width(),s.model},
            std::array<int,3>{B*S,s.model,s.inner()},std::array<int,3>{B*S,s.inner(),s.model},
            std::array<int,3>{B*S,s.model,s.width()},std::array<int,3>{s.width(),s.model,B*S},
            std::array<int,3>{s.model,s.inner(),B*S}})
            require(mp::geometry(projection_dims[0],projection_dims[1],projection_dims[2]),"Mamba3 WMMA projection geometry exceeds supported bounds");
#ifdef USE_CUDA
        require(mp::supported(p->projection_policy),"Mamba3 WMMA requires compiled RDNA3 gfx11 wave32 provider");
#else
        throw std::runtime_error("Mamba3 lowp projection requires GPU build");
#endif
    }
    p->valid=lengths.empty()?std::vector<int>(B,S):lengths;require(p->valid.size()==std::size_t(B),"Mamba3 prefix shape");for(int n:p->valid) require(n>=0&&n<=S,"Mamba3 invalid prefix");
    for(auto* param:parameters()) p->versions.push_back(param->version);
#ifndef USE_CUDA
    require(layer.device==Device::CPU,"Mamba3 GPU requires GPU build; no host fallback");
#else
    if(layer.device==Device::GPU) {checked(cudaGetDevice(&p->device_id),"Mamba3 selected device");p->stream=gpu::current_stream();require(p->prefixes.ensure(B),"Mamba3 prefix allocation");checked(cudaMemcpy(p->prefixes.get(),p->valid.data(),B*sizeof(int),cudaMemcpyHostToDevice),"Mamba3 prefix upload");record_gpu_transfer(Device::GPU,Device::CPU,B*sizeof(int));}
#endif
    p->in_weight=layer.params[0].data.clone();p->out_weight=layer.params[1].data.clone();
    // ModelSerializer stages/replaces individual Parameter storage on load.
    // Repack from the LIVE canonical registry, never from a stale packed owner.
    p->core=flat(mb::Layout(s).total,layer.device);
    for(std::size_t i=0;i<p->core_fields.size();++i) {const auto& f=p->core_fields[i];p->core.storage_view(f.offset,f.shape).copy_from(layer.params[i+2].data);}
    p->initial=empty(initial)?zero(s,layer.device):clone(initial);p->final=zero(s,layer.device);p->status=Tensor::zeros({B},layer.device);
    p->input=Tensor::zeros({B*S,layer.model},layer.device);p->mask(input,p->input,layer.model);
    // Weight validation reads all values even for an entirely masked batch.
    for(const auto* weight:{&p->in_weight,&p->out_weight}) {
        if(layer.device==Device::CPU) {bool bad=false;for(std::size_t i=0;i<std::size_t(weight->size);++i) bad|=!std::isfinite(weight->data()[i]);if(bad) for(int b=0;b<B;++b) p->status.data()[b]=2;}
#ifdef USE_CUDA
        else require(mb::gpu_parameter_check(B,std::size_t(weight->size),weight->raw_data(),p->status.raw_data()),"Mamba3 weight preflight enqueue failed");
#endif
    }
    p->projection=p->dense(p->input,p->in_weight,false,true,mp::ExactAxis::OutputCols,2*s.inner()+2*s.bc());p->mixed=Tensor::zeros({B*S,s.inner()},layer.device);
    if(p->gpu_provider!=mb::GpuProvider::DenseReference) p->coefficients=flat(mb::coefficient_size(s),layer.device);
    if(mb::is_hierarchical_provider(p->gpu_provider)) {p->hierarchy_a=flat(mb::hierarchy_elements(s),layer.device);p->hierarchy_b=flat(mb::hierarchy_elements(s),layer.device);}
    p->history=flat(mb::is_flash_provider(p->gpu_provider)?mb::checkpoint_history_size(s):mb::history_size(s),layer.device);p->q=flat(mb::rotation_size(s),layer.device);p->k=flat(mb::rotation_size(s),layer.device);p->phase=flat(mb::phase_size(s),layer.device);p->readout=flat(mb::readout_size(s),layer.device);
    if(layer.device==Device::CPU) {for(int b=0;b<B;++b) if(!p->status.data()[b]) mb::detail::forward_batch(s,mb::Layout(s),b,p->projection.data(),p->core.data(),p->valid.data(),source(p->initial),dest(p->final),p->trace(),p->mixed.data(),p->status.data());++layer.telemetry.cpu_forward;}
#ifdef USE_CUDA
    else {require(mb::gpu_forward(s,p->projection.raw_data(),p->core.raw_data(),p->prefixes.get(),source(p->initial),dest(p->final),p->trace(),p->mixed.raw_data(),p->status.raw_data()),"Mamba3 integral forward enqueue failed; discard/drain lane");++layer.telemetry.gpu_forward;
        switch(p->gpu_provider) {
        case mb::GpuProvider::DenseReference: ++layer.telemetry.gpu_reference_forward;break;
        case mb::GpuProvider::ParallelFp32: ++layer.telemetry.gpu_parallel_forward;break;
        case mb::GpuProvider::FlashFp32HierarchicalV1:
        case mb::GpuProvider::FlashFp32ReplayLdsV2:
        case mb::GpuProvider::FlashFp32: ++layer.telemetry.gpu_flash_forward;break;
        }
    }
#endif
    p->gate(p->mixed);p->output=p->dense(p->mixed,p->out_weight,false,true);p->check(p->output,layer.model);p->gate(p->output);p->gate_state(p->final);
    layer.telemetry.projection_wmma_gemms+=p->projection_dispatches;
    layer.telemetry.peak_workspace_bytes=std::max(layer.telemetry.peak_workspace_bytes,p->workspace());return std::shared_ptr<Mamba3Tape>(new Mamba3Tape(std::move(p)));
}
Mamba3Backward Mamba3Layer::backward_owned(const std::shared_ptr<Mamba3Tape>& tape,const Tensor& dy,const Mamba3State& seed) {
    require(bool(tape)&&tape->impl_->model_id==impl_->id,"Mamba3 foreign/null tape");const auto before=tape->impl_->projection_dispatches;auto result=tape->backward(dy,seed);
    impl_->telemetry.projection_wmma_gemms+=tape->impl_->projection_dispatches-before;
    if(impl_->device==Device::CPU) ++impl_->telemetry.cpu_backward;else {
        ++impl_->telemetry.gpu_backward;
        // Do not resample the environment between forward and its owned VJP.
        switch(tape->impl_->gpu_provider) {
        case mb::GpuProvider::DenseReference: ++impl_->telemetry.gpu_reference_backward;break;
        case mb::GpuProvider::ParallelFp32: ++impl_->telemetry.gpu_parallel_backward;break;
        case mb::GpuProvider::FlashFp32HierarchicalV1:
        case mb::GpuProvider::FlashFp32ReplayLdsV2:
        case mb::GpuProvider::FlashFp32: ++impl_->telemetry.gpu_flash_backward;break;
        }
    }
    impl_->telemetry.peak_workspace_bytes=std::max(impl_->telemetry.peak_workspace_bytes,tape->workspace_bytes());return result;
}
void Mamba3Layer::publish(Mamba3Backward& result) {
    require(bool(result.owner)&&result.owner->impl_->model_id==impl_->id,"Mamba3 gradient owner mismatch");auto& tape=*result.owner->impl_;tape.lane();
    require(tape.consumed&&!tape.published&&!tape.cancelled,"Mamba3 gradient publication reused/cancelled/not computed");
    auto p=parameters();require(result.parameters.size()==p.size(),"Mamba3 gradient registry mismatch");
    for(std::size_t i=0;i<p.size();++i) {require(p[i]->version==tape.versions[i],"Mamba3 weights changed since forward");tensor_shape(result.parameters[i],p[i]->data.shape.dims,p[i]->data.get_device());}
    for(int code:result.owner->audit_status()) require(code==0,"Mamba3 numeric status rejects gradient publication");
    tape.published=true;for(std::size_t i=0;i<p.size();++i) p[i]->add_grad(result.parameters[i]);
}
Tensor Mamba3Layer::forward(const Tensor& input,Context* context,const std::vector<int>& valid) {
    if(context&&context->abort_signal&&context->abort_signal->load()) throw AbortException();
    require(!impl_->pending,"Mamba3 pending training tape must be backwarded/cancelled before next forward");
    require(!(impl_->training&&impl_->streaming),"Mamba3 implicit streaming is inference-only; use owning API for BPTT");
    auto tape=forward_owned(input,impl_->streaming?impl_->session:Mamba3State{},valid);Tensor result=tape->output();
    if(impl_->streaming) {for(int code:tape->audit_status()) require(code==0,"Mamba3 failed inference must not publish session state");impl_->session=tape->snapshot_final_state();}
    if(impl_->training) impl_->pending=std::move(tape);return result;
}
Tensor Mamba3Layer::backward(const Tensor& dy,Context& context) {
    if(context.abort_signal&&context.abort_signal->load()) {cancel_pending();throw AbortException();}
    require(bool(impl_->pending),"Mamba3 backward without pending tape");auto tape=std::move(impl_->pending);auto result=backward_owned(tape,dy);publish(result);return result.input;
}
void Mamba3Layer::cancel_pending() {if(impl_->pending) {impl_->pending->cancel();impl_->pending.reset();++impl_->telemetry.cancelled;}}
void Mamba3Layer::reset() {cancel_pending();impl_->session={};}
void Mamba3Layer::to(Device dev) {require(!impl_->pending,"Mamba3 cannot move with pending backward");if(dev==impl_->device) return;
#ifndef USE_CUDA
    require(dev==Device::CPU,"Mamba3 GPU requires GPU build");
#endif
    Tensor packed=flat(mb::Layout(geometry(impl_->model,impl_->config)).total,dev);
    auto f=fields(geometry(impl_->model,impl_->config));
    for(std::size_t i=0;i<f.size();++i) packed.storage_view(f[i].offset,f[i].shape).copy_from(impl_->params[i+2].data.to(dev));
    for(auto& p:impl_->params) {p.data=p.data.to(dev);if(p.grad.size) p.grad=p.grad.to(dev);p.mark_updated();}impl_->core=std::move(packed);impl_->rebind();impl_->session=move(impl_->session,dev);impl_->device=dev;
}
void Mamba3Layer::set_training_mode(bool enabled) {require(!impl_->pending,"Mamba3 mode change with pending tape");impl_->training=enabled;}
void Mamba3Layer::set_streaming_mode(bool enabled) {if(enabled!=impl_->streaming) {reset();impl_->streaming=enabled;}}
bool Mamba3Layer::streaming_mode() const {return impl_->streaming;}
int Mamba3Layer::streaming_batch_size() const {return empty(impl_->session)?0:impl_->session.phase.shape.dims[0];}
const Mamba3Config& Mamba3Layer::config() const {return impl_->config;}
std::string Mamba3Layer::configuration_identity() const {return identity(impl_->model,impl_->config);}
const std::string& Mamba3Layer::get_layer_name() const {return impl_->name;}
void Mamba3Layer::set_layer_name(const std::string& name) {require(!name.empty(),"Mamba3 empty layer name");impl_->name=name;}
Mamba3Telemetry Mamba3Layer::telemetry() const {return impl_->telemetry;}
void Mamba3Layer::reset_runtime_telemetry() {impl_->telemetry={};}
Mamba3SessionSnapshot Mamba3Layer::snapshot_streaming_state(bool device_resident) const {return {1,configuration_identity(),impl_->streaming,clone(move(impl_->session,device_resident?impl_->device:Device::CPU))};}
void Mamba3Layer::restore_streaming_state(const Mamba3SessionSnapshot& snapshot) {
    require(!impl_->pending,"Mamba3 session restore with pending tape");require(snapshot.schema_version==1&&snapshot.configuration==configuration_identity(),"Mamba3 incompatible session schema/configuration");
    if(!empty(snapshot.state)) {const auto B=snapshot.state.phase.shape.dims.at(0);auto s=geometry(impl_->model,impl_->config,B);state_shape(snapshot.state,s,snapshot.state.phase.get_device());for(const auto* field:{&snapshot.state.phase,&snapshot.state.ssm,&snapshot.state.k,&snapshot.state.v}) {auto host=field->cpu();for(std::size_t i=0;i<std::size_t(host.size);++i) require(std::isfinite(host.data()[i]),"Mamba3 nonfinite session restore");}}
    auto staged=clone(move(snapshot.state,impl_->device));impl_->session=std::move(staged);impl_->streaming=snapshot.enabled;
}
std::vector<Mamba3SessionSnapshot> Mamba3Layer::snapshot_streaming_state_batch(bool resident) const {
    auto snapshot=snapshot_streaming_state(resident);std::vector<Mamba3SessionSnapshot> out;const int B=streaming_batch_size();
    for(int b=0;b<B;++b) {const auto& s=snapshot.state;out.push_back({1,snapshot.configuration,snapshot.enabled,{s.phase.slice(0,b,b+1).clone(),s.ssm.slice(0,b,b+1).clone(),s.k.slice(0,b,b+1).clone(),s.v.slice(0,b,b+1).clone()}});}return out;
}
void Mamba3Layer::restore_streaming_state_batch(const std::vector<Mamba3SessionSnapshot>& snapshots) {
    if(snapshots.empty()) {reset();return;}require(!impl_->pending,"Mamba3 batch restore with pending tape");
    const int B=int(snapshots.size());const auto s=geometry(impl_->model,impl_->config,B);auto staged=zero(s,impl_->device);
    for(int b=0;b<B;++b) {const auto& snapshot=snapshots[b];require(snapshot.schema_version==1&&snapshot.configuration==configuration_identity()&&snapshot.enabled==snapshots[0].enabled&&!empty(snapshot.state),"Mamba3 heterogeneous batch session metadata");state_shape(snapshot.state,geometry(impl_->model,impl_->config),snapshot.state.phase.get_device());
        auto part=move(snapshot.state,impl_->device);for(auto pair:{std::pair{&staged.phase,&part.phase},std::pair{&staged.ssm,&part.ssm},std::pair{&staged.k,&part.k},std::pair{&staged.v,&part.v}}) pair.first->storage_view(std::size_t(b)*pair.second->size,pair.second->shape.dims).copy_from(*pair.second);}
    restore_streaming_state({1,configuration_identity(),snapshots[0].enabled,staged});
}
namespace {
template<class T> void write(std::ostream& o,const T& value) {o.write(reinterpret_cast<const char*>(&value),sizeof(T));require(bool(o),"Mamba3 checkpoint write failed");}
template<class T> T read(std::istream& in) {T value{};in.read(reinterpret_cast<char*>(&value),sizeof(T));require(bool(in),"Mamba3 truncated checkpoint");return value;}
void write_string(std::ostream& o,const std::string& value) {write(o,std::uint32_t(value.size()));o.write(value.data(),value.size());require(bool(o),"Mamba3 checkpoint string write");}
std::string read_string(std::istream& in) {auto n=read<std::uint32_t>(in);require(n<=4096,"Mamba3 oversized checkpoint identity");std::string value(n,'\0');in.read(value.data(),n);require(bool(in),"Mamba3 truncated checkpoint identity");return value;}
void write_tensor(std::ostream& out,const Tensor& t) {write(out,std::uint64_t(t.size));Tensor host=t.cpu();out.write(reinterpret_cast<const char*>(host.data()),t.size*sizeof(float));require(bool(out),"Mamba3 checkpoint tensor write");}
Tensor read_tensor(std::istream& in,const std::vector<int>& dims) {Tensor t=Tensor::zeros(dims);require(read<std::uint64_t>(in)==std::uint64_t(t.size),"Mamba3 checkpoint tensor shape mismatch");in.read(reinterpret_cast<char*>(t.data()),t.size*sizeof(float));require(bool(in),"Mamba3 truncated checkpoint tensor");for(std::size_t i=0;i<std::size_t(t.size);++i) require(std::isfinite(t.data()[i]),"Mamba3 checkpoint nonfinite tensor");return t;}
}
void Mamba3Layer::save_checkpoint(const std::string& path,bool session) const {
    require(!impl_->pending,"Mamba3 checkpoint with pending training tape");const std::filesystem::path destination(path);const auto temp=checkpoint_io::unique_temporary_path(destination);
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);require(bool(out),"Mamba3 checkpoint open");write(out,std::uint64_t(0x31544b4342334d));write_string(out,configuration_identity());write(out,std::uint32_t(impl_->params.size()));
    for(const auto& p:impl_->params) {write_string(out,p.base_name);write(out,p.version);write_tensor(out,p.data);}
    const auto snap=snapshot_streaming_state(false);const int B=session&&!empty(snap.state)?snap.state.phase.shape.dims[0]:0;write(out,std::uint32_t(B));write(out,std::uint32_t(session&&snap.enabled));
    if(B) {write_tensor(out,snap.state.phase);write_tensor(out,snap.state.ssm);write_tensor(out,snap.state.k);write_tensor(out,snap.state.v);}out.flush();require(bool(out),"Mamba3 checkpoint flush");out.close();checkpoint_io::flush_file(temp);checkpoint_io::atomic_replace(temp,destination);
}
void Mamba3Layer::load_checkpoint(const std::string& path,bool restore) {
    require(!impl_->pending,"Mamba3 checkpoint load with pending tape");std::ifstream in(path,std::ios::binary);require(bool(in),"Mamba3 checkpoint open");require(read<std::uint64_t>(in)==0x31544b4342334d,"Mamba3 checkpoint magic/schema");require(read_string(in)==configuration_identity(),"Mamba3 checkpoint architecture identity mismatch");require(read<std::uint32_t>(in)==impl_->params.size(),"Mamba3 checkpoint registry count");
    std::vector<Tensor> staged;for(const auto& p:impl_->params) {require(read_string(in)==p.base_name,"Mamba3 checkpoint registry identity");read<std::uint64_t>(in);staged.push_back(read_tensor(in,p.data.shape.dims).to(impl_->device));}
    const auto B=read<std::uint32_t>(in),enabled=read<std::uint32_t>(in);require(enabled<=1,"Mamba3 checkpoint session flag");Mamba3State session;
    if(B) {const auto s=geometry(impl_->model,impl_->config,int(B));session={read_tensor(in,{s.batch,s.heads,s.rotary_pairs}),read_tensor(in,{s.batch,s.heads,s.head_dim,s.state_dim}),read_tensor(in,{s.batch,s.heads,s.rank,s.state_dim}),read_tensor(in,{s.batch,s.heads,s.head_dim})};session=move(session,impl_->device);}
    require(in.peek()==std::char_traits<char>::eof(),"Mamba3 trailing checkpoint data");
    // All metadata, payloads, devices and finiteness validated before mutation.
    // Successful load bumps live epochs rather than resurrecting stale caches.
    for(std::size_t i=0;i<staged.size();++i) {impl_->params[i].copy_data_from(staged[i]);impl_->params[i].zero_grad();}
    impl_->session=restore?clone(session):Mamba3State{};impl_->streaming=restore&&enabled;
}
}
