#include "gpu_parity_common.h"
#include "gpu_sparse_adam.h"
#include "muon_math.h"
#include "../muon_oracle_fixture.h"
#include "cuda/device_buffer.h"
#include "cuda/sparse_optimizer_activity.cuh"
#include <array>
#include <cstring>
#include <limits>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
#ifdef USE_CUDA
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Oracle {
    std::array<std::vector<float>,4> w,m,v;
    std::array<bool,4> present{};
    std::array<uint64_t,4> versions{1,1,1,1};
    Oracle(){for(int t=0;t<4;++t){w[t].resize(6);m[t].assign(6,0);v[t].assign(6,0);for(int i=0;i<6;++i)w[t][i]=1+t*.25f+i*.125f;}}
    double update(const std::array<bool,4>& active,const std::array<std::vector<float>,4>& raw,int step){
        double sum=0;for(int t=0;t<4;++t)if(active[t])for(float x:raw[t])sum+=double(x/2)*(x/2);
        const double norm=std::sqrt(sum);const float clip=norm>1?float(1/(norm+1e-6)):1;
        for(int t=0;t<4;++t)if(active[t]) {
            present[t]=true;++versions[t];auto grad=raw[t];for(auto& value:grad)value=(value*.5f)*clip;
            std::vector<float> direction;
            if(t<2)direction=nsos::muon::direction(grad,m[t],t==0?2:3,t==0?3:2);
            const float lr=t<2?.02f:.01f;
            for(int i=0;i<6;++i) {
                float delta;
                if(t<2)delta=direction[i];
                else {m[t][i]=.5f*m[t][i]+.5f*grad[i];v[t][i]=.75f*v[t][i]+.25f*grad[i]*grad[i];delta=(m[t][i]/(1-std::pow(.5f,step)))/(std::sqrt(v[t][i]/(1-std::pow(.75f,step)))+1e-8f);}
                if(t!=2)w[t][i]-=lr*.03f*w[t][i];w[t][i]-=lr*delta;
            }
        }
        return norm;
    }
};
struct Fixture {
    std::array<nsos::Parameter,4> p;
    std::shared_ptr<nsos::GpuGradientActivity> activity=std::make_shared<nsos::GpuGradientActivity>(4);
    nsos::cuda_detail::DeviceBuffer<int> offsets;
    nsos::GpuSparseAdam owner;
    Fixture(){
        Oracle oracle;const std::array<std::string,4> names{"layers.0.mamba3.in_proj.weight","layers.0.mamba3.out_proj.weight","layers.0.mamba3.D","embedding.weight"};
        for(int t=0;t<4;++t) {
            const std::vector<int> shape=t==0?std::vector<int>{2,3}:t==1?std::vector<int>{3,2}:t==2?std::vector<int>{6}:std::vector<int>{2,3};
            nsos::Tensor w(shape,nsos::Device::CPU);std::copy(oracle.w[t].begin(),oracle.w[t].end(),w.data());
            p[t]=nsos::Parameter(w.to(nsos::Device::GPU),names[t]);p[t].grad=nsos::Tensor::uninitialized(shape,nsos::Device::GPU);
            p[t].bind_device_gradient_activity(activity,t);
        }
        require(offsets.ensure(5)!=nullptr,"route allocation");
    }
    void route(const std::array<bool,4>& selected){
        std::array<int,5> host{};for(int i=0;i<4;++i)host[i+1]=host[i]+(selected[i]?1:0);
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));check(cudaMemcpy(offsets.get(),host.data(),sizeof(host),cudaMemcpyHostToDevice));
        require(launch_sparse_optimizer_activity_update(activity->view(),offsets.get(),4),"route update");
    }
    void gradients(const std::array<std::vector<float>,4>& grad){for(int t=0;t<4;++t){nsos::Tensor host(p[t].data.shape.dims,nsos::Device::CPU);std::copy(grad[t].begin(),grad[t].end(),host.data());p[t].grad.copy_from(host.to(nsos::Device::GPU));}}
    void configure(float override_lr=0){std::vector<nsos::GpuSparseAdamSlot> slots;for(int t=0;t<4;++t)slots.push_back({&p[t],override_lr?override_lr:t<2?.02f:.01f,t!=2,false,t<2?nsos::GpuSparseAlgorithm::MuonNs5Fp32:nsos::GpuSparseAlgorithm::AdamW});owner.configure(slots);}
    nsos::GpuSparseAdamOptions options(int step,bool deterministic,bool clear=true){nsos::GpuSparseAdamOptions o;o.beta1=.5f;o.beta2=.75f;o.bc1=1-std::pow(.5f,step);o.bc2=1-std::pow(.75f,step);o.max_norm=1;o.weight_decay=.03f;o.accumulation_steps=2;o.deterministic=deterministic;o.fused_epilogue=true;o.clear_gradients=clear;return o;}
    void compare(const Oracle& oracle){
        const auto state=owner.snapshot();
        for(int t=0;t<4;++t) {
            require(p[t].version==oracle.versions[t],"active/inactive version differs");
            require(state[t].initialized==oracle.present[t],"lazy presence differs");
            require(state[t].algorithm==(t<2?nsos::GpuSparseAlgorithm::MuonNs5Fp32:nsos::GpuSparseAlgorithm::AdamW),"algorithm snapshot differs");
            const auto w=p[t].data.cpu();for(int i=0;i<6;++i)require(std::abs(w.data()[i]-oracle.w[t][i])<5e-6,"mixed Muon/Adam weight oracle differs");
            if(state[t].initialized){const auto m=state[t].m.cpu(),v=state[t].v.cpu();for(int i=0;i<6;++i){require(std::abs(m.data()[i]-oracle.m[t][i])<4e-6,"momentum oracle differs");require(std::abs(v.data()[i]-oracle.v[t][i])<4e-6,"second moment oracle differs");}}
            else require(!state[t].m.size&&!state[t].v.size,"absent lazy state exposed");
        }
    }
};
std::array<std::vector<float>,4> gradients(const std::array<bool,4>& active,int step){
    std::array<std::vector<float>,4> result;
    for(int t=0;t<4;++t)for(int i=0;i<6;++i)result[t].push_back(active[t]?(step==3?0.0f:float(std::sin(i*.31+step)+.2*t)):std::numeric_limits<float>::quiet_NaN());return result;
}
void trajectory(bool deterministic) {
    Fixture fixture;Oracle oracle;
    const std::array<std::array<bool,4>,5> selection{{{{true,false,true,false}},{{false,true,false,true}},{{true,true,false,false}},{{false,false,false,false}},{{true,false,true,true}}}};
    for(int step=1;step<=5;++step) {
        fixture.activity->reset();fixture.route(selection[step-1]);const auto grad=gradients(selection[step-1],step);fixture.gradients(grad);fixture.configure();
        const auto result=fixture.owner.step(fixture.options(step,deterministic));require(result.committed,"valid hybrid group rejected");
        const double norm=oracle.update(selection[step-1],grad,step);require(std::abs(result.norm-norm)<3e-6,"mean-before-clip norm differs");fixture.compare(oracle);
        for(int t=0;t<4;++t){const auto g=fixture.p[t].grad.cpu();for(int i=0;i<6;++i)require(selection[step-1][t]?g.data()[i]==0:std::isnan(g.data()[i]),"epilogue cleared inactive or retained active gradients");}
    }
    // Ordered snapshot/restore continues both algorithms with exact lazy state.
    const auto saved=fixture.owner.snapshot();Fixture resumed;
    for(int t=0;t<4;++t){resumed.p[t].data.copy_from(fixture.p[t].data);resumed.p[t].version=fixture.p[t].version;}
    const std::array<bool,4> active{true,true,true,true};resumed.activity->reset();resumed.route(active);const auto grad=gradients(active,6);resumed.gradients(grad);resumed.configure();resumed.owner.restore(saved);
    require(resumed.owner.step(resumed.options(6,deterministic)).committed,"resumed hybrid rejected");oracle.update(active,grad,6);resumed.compare(oracle);
    // A kind mismatch may not reinterpret Adam m/v as Muon momentum.
    auto malformed=resumed.owner.snapshot();malformed[0].algorithm=nsos::GpuSparseAlgorithm::AdamW;
    bool rejected=false;try{resumed.owner.restore(malformed);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"algorithm-mismatched checkpoint accepted");resumed.compare(oracle);
    malformed=resumed.owner.snapshot();malformed[0].v=nsos::Tensor::ones(malformed[0].v.shape,nsos::Device::CPU);
    rejected=false;try{resumed.owner.restore(malformed);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Adam second moment accepted as Muon state");resumed.compare(oracle);
    // Active NaN rejects before scaling/initialization/epilogue writes.
    resumed.activity->reset();resumed.route({true,false,false,false});auto invalid=gradients({true,false,false,false},7);invalid[0][2]=std::numeric_limits<float>::quiet_NaN();resumed.gradients(invalid);resumed.configure();
    require(!resumed.owner.step(resumed.options(7,deterministic)).committed,"nonfinite active input committed");resumed.compare(oracle);
    // Recover from that group using fresh union/activity and finite gradients.
    resumed.activity->reset();resumed.route(active);resumed.gradients(grad);resumed.configure();require(resumed.owner.step(resumed.options(7,deterministic)).committed,"valid recovery rejected");oracle.update(active,grad,7);resumed.compare(oracle);
}
void cold_rollback(bool deterministic) {
    Fixture f;Oracle oracle;f.activity->reset();f.route({true,false,false,false});auto grad=gradients({true,false,false,false},1);f.gradients(grad);f.configure();require(f.owner.step(f.options(1,deterministic)).committed,"warmup failed");oracle.update({true,false,false,false},grad,1);f.compare(oracle);
    // First attempted contribution of slot1 has extreme finite LR/weights;
    // decay overflows. Roll back momentum, scaled/cleared g, logical presence,
    // weight and local versions even though NS and fused epilogue ran.
    oracle.w[1].assign(6,std::numeric_limits<float>::max());nsos::Tensor huge({3,2},nsos::Device::CPU);std::copy(oracle.w[1].begin(),oracle.w[1].end(),huge.data());f.p[1].data.copy_from(huge.to(nsos::Device::GPU));
    f.activity->reset();f.route({false,true,false,false});grad=gradients({false,true,false,false},2);f.gradients(grad);f.configure(std::numeric_limits<float>::max());
    require(!f.owner.step(f.options(2,deterministic)).committed,"update-time overflow committed");f.compare(oracle);
    const auto restored=f.p[1].grad.cpu();for(int i=0;i<6;++i)require(restored.data()[i]==grad[1][i],"rollback failed to restore cleared/scaled gradient");
}
void primary_oracle() {
    for(const auto& fixture:kMuonOracle) {
        nsos::Tensor w({fixture.rows,fixture.cols},nsos::Device::CPU),g(w.shape.dims,nsos::Device::CPU),m(w.shape.dims,nsos::Device::CPU),v(w.shape.dims,nsos::Device::CPU);
        std::copy(fixture.gradient.begin(),fixture.gradient.end(),g.data());std::copy(fixture.incoming.begin(),fixture.incoming.end(),m.data());
        nsos::Parameter p(w.to(nsos::Device::GPU),"layers.0.mamba3.in_proj.weight");p.grad=g.to(nsos::Device::GPU);
        nsos::GpuSparseAdam owner;owner.configure({{&p,1,false,true,nsos::GpuSparseAlgorithm::MuonNs5Fp32}});
        owner.restore({{p.name,p.version,true,m,v,nsos::GpuSparseAlgorithm::MuonNs5Fp32}});
        nsos::GpuSparseAdamOptions options;options.bc1=.1f;options.bc2=.001f;options.max_norm=1e4f;
        require(owner.step(options).committed,"primary oracle group rejected");const auto actual=p.data.cpu(),momentum=owner.snapshot()[0].m.cpu();
        for(size_t i=0;i<fixture.direction.size();++i){require(std::abs(actual.data()[i]+fixture.direction[i])<5e-5,"GPU primary pinned FP32-adapter direction differs");require(std::abs(momentum.data()[i]-fixture.momentum[i])<3e-7,"GPU primary momentum differs");}
    }
}
void fused_reset_proof() {
    nsos::Parameter p(nsos::Tensor::ones({2,3},nsos::Device::GPU),"layers.0.mamba3.in_proj.weight");
    p.grad=nsos::Tensor::ones({2,3},nsos::Device::GPU);
    nsos::GpuSparseAdam owner;owner.configure({{&p,.02f,false,true,nsos::GpuSparseAlgorithm::MuonNs5Fp32}});
    nsos::GpuSparseAdamOptions o;o.bc1=.1f;o.bc2=.001f;o.fused_epilogue=true;o.clear_gradients=true;
    require(owner.step(o).committed,"fused reset fixture failed");
    require(owner.consume_fused_gradient_reset({&p}),"committed clear proof unavailable");
    require(!owner.consume_fused_gradient_reset({&p}),"fused reset proof reused twice");
    p.grad.copy_from(nsos::Tensor::ones({2,3},nsos::Device::GPU));owner.configure({{&p,.02f,false,true,nsos::GpuSparseAlgorithm::MuonNs5Fp32}});
    require(owner.step(o).committed,"second fused fixture failed");const auto state=owner.snapshot();owner.restore(state);
    require(!owner.consume_fused_gradient_reset({&p}),"restore retained a clear proof");
    p.grad.copy_from(nsos::Tensor::ones({2,3},nsos::Device::GPU));owner.configure({{&p,.02f,false,true,nsos::GpuSparseAlgorithm::MuonNs5Fp32}});owner.abort();
    require(!owner.consume_fused_gradient_reset({&p}),"abort retained a clear proof");
}
#endif
}
int main()try {
    nsos::gpu_parity_test::require_cuda_device("muon_transaction");
#ifdef USE_CUDA
    primary_oracle();trajectory(true);trajectory(false);cold_rollback(true);cold_rollback(false);fused_reset_proof();
#endif
    std::cout<<"GPU Muon: pinned oracle, hybrid Adam noDecay D, rectangular/Nesterov, activity/lazy/zero/empty, fused clearing, resume-kind validation, finite reject/recovery and cold rollback passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
