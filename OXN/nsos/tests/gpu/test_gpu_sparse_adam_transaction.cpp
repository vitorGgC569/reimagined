#include "gpu_parity_common.h"
#include "gpu_sparse_adam.h"
#include "cuda/device_buffer.h"
#include "cuda/sparse_optimizer_activity.cuh"
#include "cuda/moe_training_kernels.cuh"
#include <array>
#include <limits>
#include <cstring>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
#ifdef USE_CUDA
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Oracle {
    std::array<std::array<double,6>,4> w{},m{},v{};
    std::array<bool,4> initialized{};
    std::array<uint64_t,4> versions{1,1,1,1};
    Oracle(){for(int e=0;e<4;++e)for(int i=0;i<6;++i)w[e][i]=1.0+e*0.25+i*0.125;}
    double update(const std::array<bool,4>& active,const std::array<std::array<float,6>,4>& g,int step,int accumulation,double maxnorm){
        double square=0;for(int e=0;e<4;++e)if(active[e])for(float x:g[e])square+=std::pow(double(x)/accumulation,2);
        const double norm=std::sqrt(square),clip=norm>maxnorm?maxnorm/(norm+1e-6):1;
        for(int e=0;e<4;++e)if(active[e]){
            initialized[e]=true;++versions[e];
            for(int i=0;i<6;++i){
                const double gradient=double(g[e][i])/accumulation*clip;
                m[e][i]=0.5*m[e][i]+0.5*gradient;v[e][i]=0.75*v[e][i]+0.25*gradient*gradient;
                const double mhat=m[e][i]/(1-std::pow(0.5,step)),vhat=v[e][i]/(1-std::pow(0.75,step));
                w[e][i]-=0.01*0.03*w[e][i];w[e][i]-=0.01*mhat/(std::sqrt(vhat)+1e-8);
            }
        }
        return norm;
    }
};
struct Fixture {
    std::array<nsos::Parameter,4> p;
    std::shared_ptr<nsos::GpuGradientActivity> activity=std::make_shared<nsos::GpuGradientActivity>(4);
    nsos::cuda_detail::DeviceBuffer<int> offsets;
    nsos::GpuSparseAdam adam;
    Fixture(){
        Oracle oracle;
        for(int e=0;e<4;++e){
            nsos::Tensor w({2,3},nsos::Device::CPU);for(int i=0;i<6;++i)w.data()[i]=float(oracle.w[e][i]);
            p[e]=nsos::Parameter(w.to(nsos::Device::GPU),"expert_"+std::to_string(e));
            p[e].grad=nsos::Tensor::uninitialized({2,3},nsos::Device::GPU);
            p[e].bind_device_gradient_activity(activity,e);
        }
        require(offsets.ensure(5)!=nullptr,"offset allocation");
    }
    void route(const std::array<bool,4>& active){
        std::array<int,5> host{};for(int e=0;e<4;++e)host[e+1]=host[e]+(active[e]?1:0);
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));
        check(cudaMemcpy(offsets.get(),host.data(),sizeof(host),cudaMemcpyHostToDevice));
        require(launch_sparse_optimizer_activity_update(activity->view(),offsets.get(),4),"route update");
    }
    void gradients(const std::array<std::array<float,6>,4>& values){
        for(int e=0;e<4;++e){nsos::Tensor host({2,3},nsos::Device::CPU);std::copy(values[e].begin(),values[e].end(),host.data());p[e].grad.copy_from(host.to(nsos::Device::GPU));}
    }
    void configure(float lr=0.01f){std::vector<nsos::GpuSparseAdamSlot> slots;for(auto& parameter:p)slots.push_back({&parameter,lr,true,false});adam.configure(slots);}
    nsos::GpuSparseAdamOptions options(int step){nsos::GpuSparseAdamOptions o;o.beta1=0.5f;o.beta2=0.75f;o.bc1=1-std::pow(0.5f,step);o.bc2=1-std::pow(0.75f,step);o.weight_decay=0.03f;o.max_norm=1;o.accumulation_steps=2;return o;}
    void compare(const Oracle& oracle){
        const auto states=adam.snapshot();
        for(int e=0;e<4;++e){
            const auto host=p[e].data.cpu();require(p[e].version==oracle.versions[e],"inactive/active version differs");
            require(states[e].initialized==oracle.initialized[e],"lazy moment presence differs");
            require(states[e].initialized || (!states[e].m.size && !states[e].v.size),"absent moment exported storage");
            for(int i=0;i<6;++i)require(std::isfinite(host.data()[i])&&std::abs(host.data()[i]-oracle.w[e][i])<3e-6,"Adam weight FP64 oracle differs");
            if(oracle.initialized[e]){const auto m=states[e].m.cpu(),v=states[e].v.cpu();for(int i=0;i<6;++i){
                require(std::abs(m.data()[i]-oracle.m[e][i])<3e-6,"Adam m FP64 oracle differs");
                require(std::abs(v.data()[i]-oracle.v[e][i])<3e-6,"Adam v FP64 oracle differs");}}
        }
    }
};
void trajectory(bool deterministic){
    Fixture f;Oracle oracle;
    const std::array<std::array<bool,4>,5> active{{{{true,false,false,false}},{{false,true,false,false}},{{true,false,true,false}},{{false,false,false,false}},{{true,false,false,false}}}};
    for(int step=1;step<=5;++step){
        f.activity->reset();f.route(active[step-1]);
        std::array<std::array<float,6>,4> g{};
        for(int e=0;e<4;++e)for(int i=0;i<6;++i)g[e][i]=active[step-1][e]?(step==3?0.0f:float(i+1)*0.25f):std::numeric_limits<float>::quiet_NaN();
        f.gradients(g);f.configure();auto options=f.options(step);options.deterministic=deterministic;
        const auto result=f.adam.step(options);require(result.committed,"valid group rejected");
        const double norm=oracle.update(active[step-1],g,step,2,1);require(std::abs(result.norm-norm)<2e-6,"norm oracle differs");f.compare(oracle);
    }
    // Export/restore is presence-aware and continues with identical global BC.
    auto saved=f.adam.snapshot();Fixture resumed;
    for(int e=0;e<4;++e){resumed.p[e].data.copy_from(f.p[e].data);resumed.p[e].version=f.p[e].version;}
    resumed.activity->reset();resumed.route({false,true,true,false});resumed.configure();resumed.adam.restore(saved);
    std::array<std::array<float,6>,4> g{};for(int e=0;e<4;++e)for(int i=0;i<6;++i)g[e][i]=(e==1||e==2)?0.5f:std::numeric_limits<float>::quiet_NaN();
    resumed.gradients(g);auto options=resumed.options(6);options.deterministic=deterministic;
    require(resumed.adam.step(options).committed,"resumed group rejected");oracle.update({false,true,true,false},g,6,2,1);resumed.compare(oracle);
    // Update-time overflow must roll back weights, moments, and lazy presence.
    resumed.activity->reset();resumed.route({false,false,false,true});g[3].fill(std::numeric_limits<float>::max()/2);resumed.gradients(g);
    resumed.configure(std::numeric_limits<float>::max());options=resumed.options(7);options.deterministic=deterministic;options.max_norm=std::numeric_limits<float>::max();
    require(!resumed.adam.step(options).committed,"overflow unexpectedly committed");resumed.compare(oracle);
}
void union_and_invalid(){
    Fixture f;f.activity->reset();f.route({true,false,false,false});f.route({false,true,false,false});
    std::array<std::array<float,6>,4> g{};g[0].fill(0);g[1].fill(0);g[2].fill(std::numeric_limits<float>::quiet_NaN());g[3]=g[2];f.gradients(g);f.configure();
    require(f.adam.step(f.options(1)).committed,"zero union rejected");const auto states=f.adam.snapshot();
    require(states[0].initialized&&states[1].initialized&&!states[2].initialized&&!states[3].initialized,"microbatch union/zero presence differs");
    f.activity->reset();const std::array<int,5> invalid{0,0,1,0,2};
    check(cudaMemcpy(f.offsets.get(),invalid.data(),sizeof(invalid),cudaMemcpyHostToDevice));
    require(launch_sparse_optimizer_activity_update(f.activity->view(),f.offsets.get(),4),"invalid route launch rejected host");
    f.configure();require(!f.adam.step(f.options(2)).committed,"invalid device routing accepted");
    require(f.p[0].version==2&&f.p[1].version==2&&f.p[2].version==1,"abort advanced version");
}

void contribution_commit(){
    Fixture f;
    nsos::cuda_detail::DeviceBuffer<nsos::GpuMoeGradientView> views;
    std::array<nsos::GpuMoeGradientView,4> host_views{};
    std::array<nsos::Tensor,4> bias,magnitude;
    const float nan=std::numeric_limits<float>::quiet_NaN();
    std::array<std::array<float,6>,4> initial{};for(auto& row:initial)row.fill(nan);f.gradients(initial);
    for(int e=0;e<4;++e){
        nsos::Tensor poison({2},nsos::Device::CPU);poison.data()[0]=nan;poison.data()[1]=nan;
        bias[e]=poison.to(nsos::Device::GPU);magnitude[e]=poison.to(nsos::Device::GPU);
        host_views[e]={f.p[e].grad.raw_data(),bias[e].raw_data(),magnitude[e].raw_data(),0};
    }
    require(views.ensure(4)!=nullptr,"commit descriptors");
    check(cudaMemcpy(views.get(),host_views.data(),sizeof(host_views),cudaMemcpyHostToDevice));
    std::array<double,4> expected{};
    const std::array<std::array<bool,4>,3> active{{{{true,false,false,false}},{{false,true,true,false}},{{true,true,false,false}}}};
    const std::array<float,3> values{8,0,4};
    for(int mb=0;mb<3;++mb){
        f.route(active[mb]);nsos::Tensor weight({4,6},nsos::Device::CPU),vector({4,2},nsos::Device::CPU);
        for(int e=0;e<4;++e){
            const float x=active[mb][e]?values[mb]:nan;
            for(int i=0;i<6;++i)weight.data()[e*6+i]=x;
            for(int i=0;i<2;++i)vector.data()[e*2+i]=x;
            if(active[mb][e])expected[e]+=double(x)*0.25;
        }
        const auto w=weight.to(nsos::Device::GPU),v=vector.to(nsos::Device::GPU);
        require(launch_moe_training_accumulate_gradients_activity(views.get(),f.activity->view(),w.raw_data(),v.raw_data(),v.raw_data(),6,2,0.25f),"contribution commit launch");
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));
    }
    for(int e=0;e<4;++e){
        const auto w=f.p[e].grad.cpu(),b=bias[e].cpu(),m=magnitude[e].cpu();
        for(int i=0;i<6;++i)require(e==3?std::isnan(w.data()[i]):w.data()[i]==expected[e],"first/add/inactive/unscale differs");
        for(int i=0;i<2;++i)require(e==3?(std::isnan(b.data()[i])&&std::isnan(m.data()[i])):(b.data()[i]==expected[e]&&m.data()[i]==expected[e]),"bias/magnitude commit differs");
    }
    f.activity->abort();nsos::Tensor bad({4,6},nsos::Device::CPU);std::fill_n(bad.data(),bad.size,nan);const auto source=bad.to(nsos::Device::GPU);
    require(launch_moe_training_accumulate_gradients_activity(views.get(),f.activity->view(),source.raw_data(),source.raw_data(),source.raw_data(),6,2),"abort commit launch");
    const auto unchanged=f.p[0].grad.cpu();for(int i=0;i<6;++i)require(unchanged.data()[i]==expected[0],"abort wrote gradient");
}

void active_qat_union(){
    Fixture f;f.route({true,false,false,false});f.route({false,true,false,false});
    std::array<std::array<float,6>,4> g{};for(auto& row:g)row.fill(std::numeric_limits<float>::quiet_NaN());g[0].fill(0);g[1].fill(0);f.gradients(g);
    auto latent=nsos::Tensor::uninitialized({4,6},nsos::Device::GPU);
    nsos::Tensor host({4,6},nsos::Device::CPU);
    const std::array<float,6> values{-2,-0.25f,0,0.25f,1,2};
    for(int e=0;e<4;++e)for(int i=0;i<6;++i)host.data()[e*6+i]=e<2?values[i]:std::numeric_limits<float>::quiet_NaN();
    latent.copy_from(host.to(nsos::Device::GPU));
    auto effective=latent.clone(),scales=nsos::Tensor::uninitialized({4},nsos::Device::GPU),loss=nsos::Tensor({1},nsos::Device::GPU);
    nsos::cuda_detail::DeviceBuffer<nsos::GpuMoeTrainingLinearView> views;nsos::cuda_detail::DeviceBuffer<nsos::GpuMoeGradientView> gradients;
    nsos::cuda_detail::DeviceBuffer<int> offsets;nsos::cuda_detail::DeviceBuffer<float> partials;nsos::cuda_detail::DeviceBuffer<double> loss_partials;
    std::array<nsos::GpuMoeTrainingLinearView,4> hv{};std::array<nsos::GpuMoeGradientView,4> hg{};
    for(int e=0;e<4;++e){hv[e].latent_weight=latent.raw_data()+e*6;hv[e].weight=effective.raw_data()+e*6;hv[e].qat_scale=scales.raw_data()+e;hv[e].inputs=3;hv[e].outputs=2;hg[e].weight=f.p[e].grad.raw_data();}
    require(views.ensure(4)&&gradients.ensure(4)&&offsets.ensure(6)&&partials.ensure(4)&&loss_partials.ensure(4),"QAT buffers");
    check(cudaMemcpy(views.get(),hv.data(),sizeof(hv),cudaMemcpyHostToDevice));check(cudaMemcpy(gradients.get(),hg.data(),sizeof(hg),cudaMemcpyHostToDevice));
    require(launch_moe_training_active_qat_regularization(views.get(),gradients.get(),f.activity->view(),offsets.get(),partials.get(),loss_partials.get(),loss.raw_data(),6,0.125f,3),"QAT union launch");
    double scale=0;for(float x:values)scale+=std::abs(double(x));scale=scale/6+1e-8;
    double loss_oracle=0;
    for(int e=0;e<4;++e){const auto actual=f.p[e].grad.cpu();for(int i=0;i<6;++i){
        if(e>=2){require(std::isnan(actual.data()[i]),"QAT touched inactive");continue;}
        const double quantized=values[i]>0.5*scale?scale:values[i]<-0.5*scale?-scale:0;
        const double diff=double(values[i])-quantized;loss_oracle+=0.5*0.125*diff*diff;
        require(std::abs(actual.data()[i]-3*0.125*diff)<1e-6,"QAT union FP64 gradient differs");
    }}
    require(std::abs(loss.cpu().data()[0]-loss_oracle)<1e-6,"QAT loss FP64 oracle differs");
}
void upstream_gate(){
    Fixture f;f.route({true,false,false,false});
    std::array<std::array<float,6>,4> g{};for(auto& row:g)row.fill(0);f.gradients(g);
    std::vector<nsos::GpuSparseAdamSlot> slots;for(auto& p:f.p)slots.push_back({&p,0.01f,true,false});
    f.adam.configure(slots,[](int* gate){check(cudaMemsetAsync(gate,1,sizeof(int),nsos::gpu::current_stream()));});
    require(!f.adam.step(f.options(1)).committed,"upstream status did not block Adam");f.compare(Oracle{});
}
#endif
}
int main(){return nsos::gpu_parity_test::run_parity("sparse_adam_transaction",[]{
#ifdef USE_CUDA
    trajectory(true);trajectory(false);union_and_invalid();contribution_commit();active_qat_union();upstream_gate();
#else
    throw std::runtime_error("Sparse Adam transaction requires GPU backend");
#endif
});}
