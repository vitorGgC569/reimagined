#include "gpu_parity_common.h"
#include "mamba3_layer.h"
#include "cuda/device_buffer.h"
#include <climits>
#include <limits>
#include <string>
#include <iostream>
#include <cstring>
#include <thread>
#include "gpu_execution.h"
using namespace nsos;
namespace gp=nsos::gpu_parity_test;
namespace {
void mode(const char* value) {
#ifdef _WIN32
    _putenv_s("NSOS_MAMBA3_GPU_PROVIDER",value);
#else
    setenv("NSOS_MAMBA3_GPU_PROVIDER",value,1);
#endif
}
void require(bool ok,const char* msg) {if(!ok) throw std::runtime_error(msg);}
template<class F> void rejected(F fn,const char* msg) {bool threw=false;try{fn();}catch(const std::exception&){threw=true;}require(threw,msg);}
void exact(const Tensor& x,const Tensor& y,const char* label) {
    const auto a=x.cpu(),b=y.cpu();require(a.size==b.size,"exact extent");
    for(int i=0;i<a.size;++i) if(!std::isfinite(a.data()[i])||std::memcmp(a.data()+i,b.data()+i,sizeof(float)))
        throw std::runtime_error(std::string(label)+" bit mismatch at "+std::to_string(i));
}
void fill(Tensor& x,float amp=.03f,float bias=0) {for(int i=0;i<x.size;++i) x.data()[i]=bias+amp*std::sin(.071f*(i+1));}
Mamba3State move(const Mamba3State& x,Device dev) {return {x.phase.to(dev),x.ssm.to(dev),x.k.to(dev),x.v.to(dev)};}
void close(const Mamba3State& x,const Mamba3State& y) {
    gp::assert_close(x.phase,y.phase,5e-4f,"phase",5e-4f);gp::assert_close(x.ssm,y.ssm,5e-4f,"ssm",5e-4f);
    gp::assert_close(x.k,y.k,5e-4f,"previousK",5e-4f);gp::assert_close(x.v,y.v,5e-4f,"rawPreviousV",5e-4f);
}
void finitecheck_direct_regression() {
#ifdef USE_CUDA
    const int B=5,S=33,W=129,total=B*S*W;
    const std::vector<int> valid{0,1,31,32,33};
    cuda_detail::DeviceBuffer<int> prefixes;require(prefixes.ensure(B)!=nullptr,"finitecheck prefix allocation");
    auto upload=[&](const std::vector<int>& lengths) {require(cudaMemcpy(prefixes.get(),lengths.data(),B*sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess,"finitecheck prefix upload");};
    auto poison_padding=[&](const std::vector<int>& lengths) {
        Tensor host=Tensor::zeros({B,S,W});
        for(int b=0;b<B;++b) for(int t=0;t<S;++t) if(lengths[b]<0||lengths[b]>S||t>=lengths[b])
            for(int i=0;i<W;++i) host.data()[(b*S+t)*W+i]=std::numeric_limits<float>::quiet_NaN();
        return host;
    };
    auto run_check=[&](const std::vector<int>& lengths,const Tensor& host,const std::vector<int>& initial,const std::vector<int>& expected) {
        upload(lengths);Tensor values=host.to(Device::GPU),hs=Tensor::zeros({B});
        for(int b=0;b<B;++b) hs.data()[b]=float(initial[b]);Tensor status=hs.to(Device::GPU);
        require(mamba3_block::gpu_check(B,S,W,prefixes.get(),values.raw_data(),status.raw_data()),"direct finitecheck launch failed");
        gp::cuda_sync_or_throw("direct finitecheck");auto got=status.cpu();
        for(int b=0;b<B;++b) require(got.data()[b]==expected[b],"direct finitecheck prefix/poison/sticky mismatch");
    };
    const std::vector<int> zero(B,0),sticky{1,2,3,0,0};Tensor host=poison_padding(valid);
    run_check(valid,host,zero,zero);run_check(valid,host,sticky,sticky);
    Tensor bad=host.clone();bad.data()[S*W]=std::numeric_limits<float>::quiet_NaN();auto expected=zero;expected[1]=3;
    run_check(valid,bad,zero,expected);
    bad=host.clone();bad.data()[total-1]=std::numeric_limits<float>::infinity();expected=zero;expected[4]=3;
    run_check(valid,bad,zero,expected);
    const std::vector<int> invalid{0,1,-1,S+1,33};expected=zero;expected[2]=expected[3]=1;
    run_check(invalid,poison_padding(invalid),zero,expected);
    for(int n:{129,257}) for(int poison:{-1,0,128,256}) {
        if(poison>=n) continue;Tensor hp=Tensor::ones({n}),hs=Tensor::zeros({B});
        if(poison>=0) hp.data()[poison]=std::numeric_limits<float>::quiet_NaN();
        for(int b=0;b<B;++b) hs.data()[b]=float(sticky[b]);auto params=hp.to(Device::GPU),status=hs.to(Device::GPU);
        require(mamba3_block::gpu_parameter_check(B,n,params.raw_data(),status.raw_data()),"direct parameter finitecheck launch failed");
        gp::cuda_sync_or_throw("direct parameter finitecheck");auto got=status.cpu();
        for(int b=0;b<B;++b) require(got.data()[b]==float(poison<0?sticky[b]:3),"parameter finitecheck did not poison whole batch or preserve sticky status");
    }
    Tensor small=Tensor::zeros({B},Device::GPU);upload(valid);
    require(!mamba3_block::gpu_check(INT_MAX,INT_MAX,2,prefixes.get(),small.raw_data(),small.raw_data()),"finitecheck overflow shape accepted");
    require(!mamba3_block::gpu_check(B,S,W,nullptr,small.raw_data(),small.raw_data()),"finitecheck null prefix accepted");
    require(!mamba3_block::gpu_parameter_check(B,std::size_t(INT_MAX)+1,small.raw_data(),small.raw_data()),"parameter finitecheck oversized count accepted");
    gp::cuda_sync_or_throw("finitecheck rejected shapes");
    std::cout<<"direct finitechecks B5/S33/W129 poison/prefix/sticky/parameter tails/overflow PASS\n";
#else
    throw std::runtime_error("Direct finitechecks require GPU build");
#endif
}
void phase_seam_regression() {
    const int B=1,S=4,D=4,H=4,P=2,N=128,A=32;
    Mamba3Config c;c.head_dim=P;c.state_dim=N;c.expand=2;c.n_groups=2;c.seed=90210;
    const float tau=float(6.283185307179586476925286766559),initial_phase=std::nextafter(tau,0.f);
    const float increment=(tau-initial_phase)*.3f,dt=std::log(2.f),angle=std::atanh(increment/(float(3.14159265358979323846)*dt));
    mode("dense_reference");Mamba3Layer cpu(D,c);auto params=cpu.parameters();
    Tensor win=Tensor::zeros(params[0]->data.shape.dims);
    const int width=params[0]->data.shape.dims[0];
    for(int a=0;a<A;++a) for(int d=0;d<D;++d) win.data()[(width-A+a)*D+d]=angle/D;
    params[0]->copy_data_from(win);params[4]->copy_data_from(Tensor::zeros({H}));
    Tensor input=Tensor::ones({B,S,D}),dy=Tensor::zeros({B,S,D});
    Mamba3State initial{Tensor::ones({B,H,A}).mul(initial_phase),Tensor::zeros({B,H,P,N}),Tensor::zeros({B,H,1,N}),Tensor::zeros({B,H,P})};
    Mamba3State seed{Tensor::ones({B,H,A}),Tensor::zeros({B,H,P,N}),Tensor::zeros({B,H,1,N}),Tensor::zeros({B,H,P})};
    auto ct=cpu.forward_owned(input,initial);const auto expected=ct->snapshot_final_state();
    for(int i=0;i<expected.phase.size;++i) require(expected.phase.data()[i]==initial_phase,"seam fixture failed to retain sub-ULP reference increments");
    auto cg=cpu.backward_owned(ct,dy,seed);
    for(const char* provider:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2"}) {
        mode(provider);Mamba3Layer gpu(D,c);auto gpu_params=gpu.parameters();
        for(std::size_t i=0;i<gpu_params.size();++i) gpu_params[i]->copy_data_from(params[i]->data);
        gpu.to(Device::GPU);auto gt=gpu.forward_owned(input.to(Device::GPU),move(initial,Device::GPU));
        const auto state=gt->snapshot_final_state();close(state,expected);
        auto phase=state.phase.cpu();for(int i=0;i<phase.size;++i) require(phase.data()[i]==initial_phase,"parallel phase reassociated sub-ULP seam increments");
        auto gg=gpu.backward_owned(gt,dy.to(Device::GPU),move(seed,Device::GPU));close(gg.initial_state,cg.initial_state);
        gp::assert_close(gg.input,cg.input,5e-4f,"seam dinput",5e-4f);
        for(std::size_t i=0;i<gg.parameters.size();++i) gp::assert_close(gg.parameters[i],cg.parameters[i],5e-4f,gpu_params[i]->base_name.c_str(),5e-4f);
        gpu.publish(gg);std::cout<<provider<<" phase seam stored state and all VJPs PASS\n";
    }
    mode("dense_reference");
}
void case_parity(int S,int R,bool norm,int model=4,int head_dim=2,int groups=2,int state_dim=128,bool full_prefix=false,float rope=.5f,bool all_empty=false) {
    const int B=2,H=model*2/head_dim,P=head_dim,N=state_dim,A=int(N*rope)/2;
    Mamba3Config c;c.head_dim=P;c.state_dim=N;c.expand=2;c.n_groups=groups;c.mimo=R>1;c.mimo_rank=R;c.outproj_norm=norm;c.rope_fraction=rope;c.seed=90210;
    mode("dense_reference");Mamba3Layer cpu(model,c);Tensor x({B,S,model}),dy({B,S,model});fill(x);fill(dy,.01f);
    const std::vector<int> lengths=all_empty?std::vector<int>{0,0}:full_prefix?std::vector<int>{S,S==1?1:S/2+1}:std::vector<int>{S==1?1:S-1,0};
    for(int b=0;b<B;++b) for(int t=lengths[b];t<S;++t) for(int d=0;d<model;++d) {x.data()[(b*S+t)*model+d]=std::numeric_limits<float>::quiet_NaN();dy.data()[(b*S+t)*model+d]=std::numeric_limits<float>::quiet_NaN();}
    Mamba3State initial{Tensor({B,H,A}),Tensor({B,H,P,N}),Tensor({B,H,R,N}),Tensor({B,H,P})};
    Mamba3State seed{Tensor({B,H,A}),Tensor({B,H,P,N}),Tensor({B,H,R,N}),Tensor({B,H,P})};
    for(auto* t:{&initial.phase,&initial.ssm,&initial.k,&initial.v}) fill(*t,.006f,.01f);
    fill(initial.phase,.05f,6.28f);
    for(auto* t:{&seed.phase,&seed.ssm,&seed.k,&seed.v}) fill(*t,.003f);
    auto ct=cpu.forward_owned(x,initial,lengths);auto cout=ct->output();auto cstate=ct->snapshot_final_state();auto cg=cpu.backward_owned(ct,dy,seed);
    std::size_t parallel_forward_bytes=0,parallel_backward_bytes=0;
    Mamba3Backward flash_v1;
    for(const char* provider:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2"}) {
        mode(provider);Mamba3Layer gpu(model,c);gpu.to(Device::GPU);auto in=move(initial,Device::GPU),gs=move(seed,Device::GPU);
        auto gt=gpu.forward_owned(x.to(Device::GPU),in,lengths);
        gp::assert_close(gt->output(),cout,5e-4f,"optimized output",5e-4f);close(gt->snapshot_final_state(),cstate);
        if(std::string(provider)=="parallel_fp32_v1") parallel_forward_bytes=gt->workspace_bytes();
        else if(S>32) require(gt->workspace_bytes()<parallel_forward_bytes,"Flash retained dense state history");
        // Provider is captured by the owning tape; later environment changes
        // must not reinterpret its storage or reverse recurrence.
        mode("invalid_after_forward");auto gg=gpu.backward_owned(gt,dy.to(Device::GPU),gs);
        gp::assert_close(gg.input,cg.input,5e-4f,"optimized input VJP",5e-4f);close(gg.initial_state,cg.initial_state);
        require(gg.parameters.size()==cg.parameters.size(),"gradient registry extent");
        for(std::size_t i=0;i<gg.parameters.size();++i) gp::assert_close(gg.parameters[i],cg.parameters[i],5e-4f,gpu.parameters()[i]->base_name.c_str(),5e-4f);
        if(std::string(provider)=="parallel_fp32_v1") parallel_backward_bytes=gt->workspace_bytes();
        else if(S>32) require(gt->workspace_bytes()<parallel_backward_bytes,"Flash backward retained dense state history");
        if(std::string(provider)=="flash_fp32_v1") flash_v1=gg;
        if(std::string(provider)=="flash_fp32_replay_lds_v2") {
            exact(gg.input,flash_v1.input,"Flash v2 bitwise input VJP");
            exact(gg.initial_state.phase,flash_v1.initial_state.phase,"Flash v2 bitwise initial phase");
            exact(gg.initial_state.ssm,flash_v1.initial_state.ssm,"Flash v2 bitwise initial SSM");
            exact(gg.initial_state.k,flash_v1.initial_state.k,"Flash v2 bitwise initial K");
            exact(gg.initial_state.v,flash_v1.initial_state.v,"Flash v2 bitwise initial raw V");
            for(std::size_t i=0;i<gg.parameters.size();++i) exact(gg.parameters[i],flash_v1.parameters[i],"Flash v2 bitwise parameter VJP");
        }
        gpu.publish(gg);rejected([&]{gpu.publish(gg);},"duplicate publication accepted");rejected([&]{gpu.backward_owned(gt,dy.to(Device::GPU));},"consumed optimized tape reused");
        mode(provider);auto repeat=gpu.forward_owned(x.to(Device::GPU),in,lengths);auto rgrad=gpu.backward_owned(repeat,dy.to(Device::GPU),gs);
        gp::assert_close(rgrad.input,gg.input,0,"deterministic optimized dinput");
        for(std::size_t i=0;i<gg.parameters.size();++i) gp::assert_close(rgrad.parameters[i],gg.parameters[i],0,"deterministic optimized parameter VJP");
        auto cancelled=gpu.forward_owned(x.to(Device::GPU),in,lengths);cancelled->cancel();rejected([&]{cancelled->backward(dy.to(Device::GPU));},"cancelled optimized tape reused");
        std::cout<<provider<<" S="<<S<<" R="<<R<<" norm="<<norm<<" forward_bytes="<<repeat->workspace_bytes()<<" PASS\n";
    }
    mode("dense_reference");
}
void v2_failure_ownership() {
    mode("flash_fp32_replay_lds_v2");Mamba3Config c;c.head_dim=2;c.n_groups=2;c.mimo=true;c.mimo_rank=4;
    Mamba3Layer layer(4,c),foreign_layer(4,c);layer.to(Device::GPU);foreign_layer.to(Device::GPU);
    Tensor x=Tensor::ones({1,33,4}),dy=Tensor::ones({1,33,4});
    auto tape=layer.forward_owned(x.to(Device::GPU));Tensor bad_dy=dy.clone();bad_dy.data()[0]=std::numeric_limits<float>::quiet_NaN();
    auto g=layer.backward_owned(tape,bad_dy.to(Device::GPU));require(tape->audit_status()[0]!=0,"v2 nonfinite adjoint accepted");
    rejected([&]{layer.publish(g);},"v2 failed gradients published");
    for(auto* p:layer.parameters()) require(!p->has_gradient(),"v2 failed VJP contributed gradients");
    Tensor bad_x=x.clone();bad_x.data()[0]=std::numeric_limits<float>::infinity();
    auto broken=layer.forward_owned(bad_x.to(Device::GPU));require(broken->audit_status()[0]!=0,"v2 nonfinite input accepted");
    const auto out=broken->output().cpu();for(int i=0;i<out.size;++i) require(out.data()[i]==0,"v2 failed output leaked");
    auto owner=layer.forward_owned(x.to(Device::GPU));
    rejected([&]{foreign_layer.backward_owned(owner,dy.to(Device::GPU));},"v2 foreign model tape accepted");
    bool foreign_thread=false;std::thread other([&]{try{owner->output();}catch(const std::exception&){foreign_thread=true;}});other.join();
    require(foreign_thread,"v2 foreign thread accepted");
    gpu::ExecutionContext lane;{gpu::ExecutionContext::Scope scope(lane);rejected([&]{owner->audit_status();},"v2 foreign stream accepted");}
    owner->cancel();rejected([&]{layer.backward_owned(owner,dy.to(Device::GPU));},"v2 cancelled tape accepted");
    rejected([&]{layer.forward_owned(x.to(Device::GPU),{}, {-1});},"v2 invalid prefix accepted");
    Context context;std::atomic<bool> stop{true};context.abort_signal=&stop;
    const auto before=layer.telemetry();rejected([&]{layer.forward(x.to(Device::GPU),&context);},"v2 pre-forward abort accepted");
    require(layer.telemetry().gpu_forward==before.gpu_forward,"v2 aborted forward enqueued GPU work");
    mode("dense_reference");std::cout<<"Flash v2 nonfinite/status/publication/tape/thread/stream/cancel gates PASS\n";
}
}
int main() {return gp::run_parity("mamba3_parallel",[] {
    set_matmul_precision_mode(0);
    for(int S:{1,31,32,33,65,129}) for(int R:{1,4}) for(bool norm:{false,true}) case_parity(S,R,norm);
    case_parity(33,8,true);case_parity(33,8,true,4,2,2,128,false,.5f,true);finitecheck_direct_regression();phase_seam_regression();
    // Exercise long inter-chunk carries and phase suffixes beyond the small
    // tail matrix, while keeping the independent CPU VJP oracle practical.
    case_parity(257,1,true);case_parity(1025,4,true);
    // Exercise LDS P/T tails, the narrow-N path and production head widths.
    case_parity(35,4,true,10,5,2,128);
    case_parity(35,8,true,16,8,1,8);
    case_parity(33,4,true,32,64,1,128);
    case_parity(5,4,true,64,128,1,128);
    // Independent reviewer: preserve all worker cases and exercise the LAST
    // valid token, both RoPE coordinate layouts, full128 owner-lane rank8 tile,
    // and a second active ragged sequence rather than only a zero prefix.
    case_parity(5,1,false,10,5,2,128,true,1.f);
    case_parity(33,4,true,10,5,2,128,true,1.f);
    case_parity(5,8,true,64,128,1,128,true,1.f);
    case_parity(65,4,false,4,2,2,128,true,.5f);
    v2_failure_ownership();mode("dense_reference");
});}
