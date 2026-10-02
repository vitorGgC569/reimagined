#include "gpu_parity_common.h"
#include "gpu_sparse_adam.h"
#include "cuda/device_buffer.h"
#include "cuda/sparse_optimizer_activity.cuh"
#include "muon-trajectory-golden.h"
#include <array>
#include <limits>
#include <cstring>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
#ifdef USE_CUDA
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Fixture {
    std::array<nsos::Parameter,4> p;
    std::shared_ptr<nsos::GpuGradientActivity> activity=std::make_shared<nsos::GpuGradientActivity>(4);
    nsos::cuda_detail::DeviceBuffer<int> offsets;
    nsos::GpuSparseAdam optimizer;
    Fixture(){
        for(int e=0;e<4;++e){
            nsos::Tensor w({e==2?3:2,e==2?2:3},nsos::Device::CPU);
            for(int i=0;i<6;++i)w.data()[i]=float(1.0+e*.25+i*.125);
            p[e]=nsos::Parameter(w.to(nsos::Device::GPU),e%2==0?"layers.0.expert_gate_up."+std::to_string(e)+".weight":"layers.0.expert_gate_up."+std::to_string(e)+".bias");
            p[e].grad=nsos::Tensor::uninitialized(w.shape.dims,nsos::Device::GPU);p[e].bind_device_gradient_activity(activity,e);
        }
        require(offsets.ensure(5)!=nullptr,"route buffer");
    }
    void route(const std::array<bool,4>& active){
        std::array<int,5> host{};for(int e=0;e<4;++e)host[e+1]=host[e]+int(active[e]);
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));check(cudaMemcpy(offsets.get(),host.data(),sizeof(host),cudaMemcpyHostToDevice));
        require(launch_sparse_optimizer_activity_update(activity->view(),offsets.get(),4),"route launch");
    }
    void gradients(int step,const std::array<bool,4>& active){
        for(int e=0;e<4;++e){nsos::Tensor g(p[e].data.shape.dims,nsos::Device::CPU);
            for(int i=0;i<6;++i)g.data()[i]=active[e]?(step==3?0.0f:float(i+1)*.25f):std::numeric_limits<float>::quiet_NaN();
            p[e].grad.copy_from(g.to(nsos::Device::GPU));}
    }
    std::vector<nsos::GpuSparseAdamSlot> slots(float lr=1){
        std::vector<nsos::GpuSparseAdamSlot> out;
        for(int e=0;e<4;++e)out.push_back({&p[e],(e%2==0?.02f:.01f)*lr,e%2==0,false,e%2==0?nsos::GpuSparseAlgorithm::MuonNs5Fp32:nsos::GpuSparseAlgorithm::AdamW});return out;
    }
    nsos::GpuSparseAdamOptions options(int step,bool deterministic,bool fused){
        nsos::GpuSparseAdamOptions o;o.beta1=.5f;o.beta2=.75f;o.bc1=1-std::pow(.5f,step);o.bc2=1-std::pow(.75f,step);o.eps=1e-8f;o.weight_decay=.03f;o.max_norm=1;o.accumulation_steps=2;o.deterministic=deterministic;o.fused_epilogue=fused;o.clear_gradients=fused;return o;
    }
    void compare(int golden){
        const auto states=optimizer.snapshot();
        for(int e=0;e<4;++e){
            require(p[e].version==golden_version[golden][e],"golden local version");require(states[e].initialized==bool(golden_present[golden][e]),"golden lazy presence");
            require(states[e].algorithm==(e%2==0?nsos::GpuSparseAlgorithm::MuonNs5Fp32:nsos::GpuSparseAlgorithm::AdamW),"algorithm export");
            const auto w=p[e].data.cpu();for(int i=0;i<6;++i)require(std::isfinite(w.data()[i])&&std::abs(w.data()[i]-golden_w[golden][e][i])<8e-5,"FP64 weight oracle");
            if(states[e].initialized){const auto m=states[e].m.cpu(),v=states[e].v.cpu();for(int i=0;i<6;++i){require(std::abs(m.data()[i]-golden_m[golden][e][i])<8e-5,"FP64 momentum oracle");require(std::abs(v.data()[i]-golden_v[golden][e][i])<8e-5,"FP64 variance oracle");if(e%2==0)require(v.data()[i]==0,"Muon v must remain exact zero");}}
            else require(!states[e].m.size&&!states[e].v.size,"absent moment storage exported");
        }
    }
};
void trajectory(bool deterministic,bool fused){
    Fixture f;
    const std::array<std::array<bool,4>,5> cohort{{{{true,false,false,false}},{{false,true,false,false}},{{true,false,true,false}},{{false,false,false,false}},{{true,false,false,false}}}};
    for(int step=1;step<=5;++step){
        f.activity->reset();f.route(cohort[step-1]);f.gradients(step,cohort[step-1]);
        // Configuration and both microbatch union publications are read-only
        // for model weights/versions: the owner receives complete VJP gradients.
        std::array<nsos::Tensor,4> before;for(int e=0;e<4;++e)before[e]=f.p[e].data.cpu();
        f.optimizer.configure(f.slots());
        for(int e=0;e<4;++e){nsos::gpu_parity_test::assert_close(f.p[e].data,before[e],0,"configure changed weight");require(f.p[e].version==golden_version[step-1][e],"configure advanced version");}
        const auto result=f.optimizer.step(f.options(step,deterministic,fused));require(result.committed,"valid hybrid group rejected");f.compare(step);
        require(std::abs(result.norm-golden_norm[step])<3e-6,"global norm/real accumulation oracle");
        if(fused)for(int e=0;e<4;++e)if(cohort[step-1][e]){auto g=f.p[e].grad.cpu();for(int i=0;i<6;++i)require(g.data()[i]==0,"fused active gradient not cleared");}
    }
    // Freeze a portable checkpoint of optimizer moments independently of owner
    // Tensor aliases; continuation uses original local versions explicitly.
    auto saved=f.optimizer.snapshot();for(auto& s:saved)if(s.initialized){s.m=s.m.cpu().clone();s.v=s.v.cpu().clone();}
    Fixture resumed;for(int e=0;e<4;++e){resumed.p[e].data.copy_from(f.p[e].data);resumed.p[e].version=f.p[e].version;}
    resumed.activity->reset();resumed.route({false,true,true,false});resumed.gradients(6,{false,true,true,false});resumed.optimizer.configure(resumed.slots());
    // Late metadata corruption must be rejected before publishing any moments.
    auto bad=saved;bad.back().algorithm=nsos::GpuSparseAlgorithm::MuonNs5Fp32;
    bool rejected=false;try{resumed.optimizer.restore(bad);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"algorithm mismatch restore accepted");
    auto nonzero=saved;nonzero[2].v=nonzero[2].v.clone();nonzero[2].v.data()[5]=.01f;
    rejected=false;try{resumed.optimizer.restore(nonzero);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"nonzero Muon variance restored");
    resumed.optimizer.restore(saved);require(resumed.optimizer.step(resumed.options(6,deterministic,fused)).committed,"checkpoint continuation rejected");resumed.compare(6);
    // A late active Adam slot overflows; already-computed Muon updates and
    // clear-grad epilogue must all roll back, including first lazy state.
    resumed.activity->reset();resumed.route({true,false,false,true});resumed.gradients(7,{true,false,false,true});
    std::array<nsos::Tensor,4> before_g;for(int e=0;e<4;++e)before_g[e]=resumed.p[e].grad.cpu();
    auto slots=resumed.slots();slots[3].learning_rate=std::numeric_limits<float>::max();slots[3].weight_decay=true;
    resumed.optimizer.configure(slots);auto o=resumed.options(7,deterministic,fused);o.weight_decay=std::numeric_limits<float>::max();
    require(!resumed.optimizer.step(o).committed,"update overflow committed");resumed.compare(6);
    for(int e:{0,3})nsos::gpu_parity_test::assert_close(resumed.p[e].grad,before_g[e],0,"rollback gradient clearing/scaling");
}
void global_gate(){
    Fixture f;f.route({true,true,false,false});f.gradients(1,{true,true,false,false});auto g=f.p[1].grad.cpu();g.data()[5]=std::numeric_limits<float>::quiet_NaN();f.p[1].grad.copy_from(g.to(nsos::Device::GPU));
    f.optimizer.configure(f.slots());require(!f.optimizer.step(f.options(1,true,true)).committed,"late Adam NaN failed global gate");f.compare(0);
    f.activity->reset();f.route({true,true,false,false});f.gradients(1,{true,true,false,false});
    f.optimizer.configure(f.slots(),[](int* status){check(cudaMemsetAsync(status,1,sizeof(int),nsos::gpu::current_stream()));});
    require(!f.optimizer.step(f.options(1,true,true)).committed,"upstream VJP failure ignored");f.compare(0);
}
void zero_union(){
    Fixture f;f.activity->reset();f.route({true,false,false,false});f.route({false,false,true,false});f.gradients(3,{true,false,true,false});
    f.optimizer.configure(f.slots());require(f.optimizer.step(f.options(1,true,true)).committed,"zero union rejected");
    auto states=f.optimizer.snapshot();require(states[0].initialized&&states[2].initialized&&!states[1].initialized&&!states[3].initialized,"zero union lazy presence");require(f.p[0].version==2&&f.p[2].version==2&&f.p[1].version==1,"zero union versions");
}
void registry_and_closed_checkpoint_guards(){
    Fixture f;f.route({true,false,false,false});f.gradients(1,{true,false,false,false});f.optimizer.configure(f.slots());
    bool rejected=false;try{f.optimizer.snapshot();}catch(const std::logic_error&){rejected=true;}require(rejected,"open optimizer snapshot accepted");
    auto options=f.options(1,true,false);options.clear_gradients=true;
    rejected=false;try{f.optimizer.step(options);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"gradient clear without opt-in epilogue accepted");
    f.optimizer.abort();f.compare(0);
    Fixture alias;alias.p[2].data=alias.p[0].data; // tied writable storage must appear once in the canonical registry
    rejected=false;try{alias.optimizer.configure(alias.slots());}catch(const std::logic_error&){rejected=true;}require(rejected,"duplicate tied weight storage accepted");
    Fixture excluded;excluded.p[0].name="layers.0.mamba3.dt_bias";
    rejected=false;try{excluded.optimizer.configure(excluded.slots());}catch(const std::invalid_argument&){rejected=true;}require(rejected,"sensitive dt assigned Muon");
}
#endif
}
int main(){return nsos::gpu_parity_test::run_parity("muon_adam_transaction_independent",[]{
#ifdef USE_CUDA
    trajectory(true,false);trajectory(true,true);trajectory(false,false);trajectory(false,true);global_gate();zero_union();registry_and_closed_checkpoint_guards();
#else
    throw std::runtime_error("Muon transaction fixture requires GPU backend");
#endif
});}
