#include "gpu_parity_common.h"
#include "gpu_attention_training.h"
#include "gpu_sparse_adam.h"
#include "bitlinear.h"
#include "gpu_execution.h"
#include "cuda/device_buffer.h"
#include <limits>
#include <iostream>
#include <cstring>
using namespace nsos;
namespace at=nsos::attention_training;
namespace ar=nsos::attention_rdna;
namespace {
void require(bool x,const char* msg){if(!x) throw std::runtime_error(msg);}
void near(float a,double b,const char* msg){if(std::abs(a-b)>2e-4*(1+std::abs(b))) throw std::runtime_error(msg);}
Tensor upload(const std::vector<int>& shape,const std::vector<float>& values){
    Tensor x(shape);std::copy(values.begin(),values.end(),x.data());return x.to(Device::GPU);
}
template<class F> void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"expected fail-closed rejection");}
void set_policy(const char* x){
#ifdef _WIN32
    _putenv_s("NSOS_ATTN_TRAINING_PROVIDER",x);
    _putenv_s("NSOS_ATTN_TILED_TRAINING","0");_putenv_s("NSOS_ATTN_BWD_HOST","0");
#else
    setenv("NSOS_ATTN_TRAINING_PROVIDER",x,1);
    setenv("NSOS_ATTN_TILED_TRAINING","0",1);setenv("NSOS_ATTN_BWD_HOST","0",1);
#endif
}
void analytic_tape(ar::Precision precision){
    // Independent analytic oracle: Q=0 => exact uniform masked probabilities.
    // Quarter-turn split-half RoPE preserves binary-representable K exactly;
    // STE VJP equals the ordinary derivative at these rounded operands.
    const ar::Shape s{2,5,4,2,3,3,0.5f,precision};at::Config c{s,2};
    const int dim=12,kdim=6;const std::vector<int> valid{5,2};
    std::vector<float> q(2*5*dim,0),kv(2*5*2*kdim),go(q.size(),0.25f);
    for(int b=0;b<2;++b) for(int i=0;i<5;++i) for(int h=0;h<2;++h) for(int d=0;d<3;++d){
        const auto t=(b*5+i)*2*kdim+h*3+d;
        kv[t]=(i+1+h+d)*0.125f;kv[t+kdim]=(i-h+d)*0.125f;
    }
    for(int i=2;i<5;++i) for(int d=0;d<dim;++d) {q[(5+i)*dim+d]=NAN;go[(5+i)*dim+d]=NAN;}
    auto qt=upload({2,5,dim},q),kvt=upload({2,5,2*kdim},kv);
    auto cs=upload({7,1},std::vector<float>(7,0)),sn=upload({7,1},std::vector<float>(7,1));
    auto tape=at::Tape::forward(c,qt,kvt,cs,sn,valid);
    // Mutation of caller storage AND public output must not modify tape inputs/O.
    qt.copy_from(Tensor::zeros(qt.shape.dims,Device::GPU));
    Tensor public_output=tape->output();public_output.copy_from(Tensor::zeros(public_output.shape.dims,Device::GPU));
    auto result=tape->backward(upload({2,5,dim},go));
    auto out=tape->output().cpu(),dq=result.q.cpu(),dkv=result.kv.cpu();
    std::vector<double> eq(q.size()),ekv(kv.size());
    for(int b=0;b<2;++b) for(int i=0;i<valid[b];++i) for(int h=0;h<4;++h){
        const int kh=h/2,begin=std::max(0,i-2),count=i-begin+1;
        double mean[3]={0,0,0};
        for(int j=begin;j<=i;++j) for(int d=0;d<3;++d) mean[d]+=kv[(b*5+j)*2*kdim+kdim+kh*3+d]/count;
        for(int d=0;d<3;++d) near(out.data()[(b*5+i)*dim+h*3+d],mean[d],"masked output oracle");
        for(int j=begin;j<=i;++j){
            double ds=0;for(int d=0;d<3;++d) ds+=0.25*(kv[(b*5+j)*2*kdim+kdim+kh*3+d]-mean[d]);
            ds*=s.scale/count;
            for(int d=0;d<3;++d){
                eq[(b*5+i)*dim+h*3+d]+=ds*kv[(b*5+j)*2*kdim+kh*3+d];
                ekv[(b*5+j)*2*kdim+kdim+kh*3+d]+=0.25/count;
            }
        }
    }
    for(std::size_t i=0;i<eq.size();++i) near(dq.data()[i],eq[i],"RoPE dQ oracle");
    for(std::size_t i=0;i<ekv.size();++i) near(dkv.data()[i],ekv[i],"GQA dKV oracle");
    for(int x:tape->audit_status()) require(x==0,"valid tape status");
    rejects([&]{tape->backward(upload({2,5,dim},go));});
    rejects([&]{at::Tape::forward(c,qt,kvt,cs,sn,{5,6});});
    require(tape->workspace_bytes()>0,"workspace accounting");
}
std::vector<Parameter*> params(BitLinear& q,BitLinear& kv,BitLinear& o){
    auto p=q.parameters();for(auto* l:{&kv,&o}){auto x=l->parameters();p.insert(p.end(),x.begin(),x.end());}return p;
}
// Regression for the optimizer owner's mandatory merge. No Trainer, MoE
// activity domain, callback or direct write to the optimizer's gate is used.
void direct_sparse_owner_status(bool warm,bool deterministic){
    set_policy("rdna_bf16_v1");set_matmul_precision_mode(0);
    gpu::ExecutionContext context;
    gpu::ExecutionContext::Scope lane(context);
    BitLinear q(8,8,true,191),kv(8,8,true,192),o(8,8,true,193);
    for(auto* l:{&q,&kv,&o}){
        l->set_exact_linear_mode(true);l->set_reference_path(true);
        l->set_training_mode(true);l->to(Device::GPU);
    }
    std::vector<Parameter*> p;
    for(auto* a:params(q,kv,o)) if(a->trainable&&a->data.size){
        a->assign_relative_name("attention_direct_owner_"+std::to_string(p.size()));
        require(!a->has_device_gradient_activity(),"owner regression requires dense API slots");
        a->track_gradient_contributions();
        a->grad=Tensor::zeros(a->data.shape.dims,Device::GPU);
        p.push_back(a);
    }
    require(p.size()==6,"exact Q/KV/out weights and biases must participate");
    at::Provider provider(q,kv,o);
    GpuSparseAdam adam;
    const at::Config c{{1,3,2,1,4,3,0.5f,ar::Precision::BF16},0};
    const auto x=upload({1,3,8},std::vector<float>(24,0.125f));
    const auto cs=upload({3,2},std::vector<float>(6,1));
    const auto sn=upload({3,2},std::vector<float>(6,0));
    const auto dy=upload({1,3,8},std::vector<float>(24,0.0625f));
    cuda_detail::DeviceBuffer<int> audit_flag;
    require(audit_flag.ensure(1)!=nullptr,"owner ledger audit allocation");
    auto issue=[&]{
        // Independent diagnostic flag; this never touches adam's gate.
        require(cudaMemsetAsync(audit_flag.get(),0,sizeof(int),gpu::current_stream())==cudaSuccess,"owner audit clear");
        at::merge_status(p,audit_flag.get());
        int result=-1;
        require(cudaMemcpyAsync(&result,audit_flag.get(),sizeof(result),cudaMemcpyDeviceToHost,gpu::current_stream())==cudaSuccess,"owner audit download");
        require(cudaStreamSynchronize(gpu::current_stream())==cudaSuccess,"owner audit fence");
        return result;
    };
    auto finite=[](const Tensor& t){
        const auto h=t.cpu();
        for(int i=0;i<h.size;++i)require(std::isfinite(h.data()[i]),"direct owner finite operands/gradients");
    };
    auto equal=[](const Tensor& a,const Tensor& b){
        if(a.shape!=b.shape||a.size!=b.size)return false;
        if(!a.size)return true;
        const auto ah=a.cpu(),bh=b.cpu();
        return std::memcmp(ah.data(),bh.data(),size_t(ah.size)*sizeof(float))==0;
    };
    auto zero_gradients=[&]{for(auto* a:p)a->zero_grad();};
    auto microbatch=[&]{
        finite(provider.forward(x,c,cs,sn,{3}));
        finite(provider.backward(dy));
    };
    auto configure=[&]{
        std::vector<GpuSparseAdamSlot> slots;
        for(auto* a:p)slots.push_back({a,0.01f,true,a->has_gradient()});
        adam.configure(slots); // Intentionally NO upstream status callback.
    };
    auto options=[&](int step){
        GpuSparseAdamOptions result;
        result.beta1=0.5f;result.beta2=0.75f;
        result.bc1=1-std::pow(result.beta1,step);result.bc2=1-std::pow(result.beta2,step);
        result.weight_decay=0.03f;result.max_norm=1;result.accumulation_steps=2;
        result.deterministic=deterministic;return result;
    };
    auto snapshot_states=[&]{
        auto states=adam.snapshot();
        for(auto& s:states)if(s.initialized){s.m=s.m.cpu().clone();s.v=s.v.cpu().clone();}
        return states;
    };
    auto same_states=[&](const std::vector<GpuSparseAdamState>& before){
        const auto after=adam.snapshot();require(after.size()==before.size(),"owner moment registry changed");
        for(size_t i=0;i<before.size();++i){
            require(after[i].name==before[i].name&&after[i].version==before[i].version&&
                    after[i].initialized==before[i].initialized,"owner moment presence/version changed on rejection");
            require(equal(after[i].m,before[i].m)&&equal(after[i].v,before[i].v),"owner moments changed on rejection");
        }
    };

    zero_gradients();configure();adam.abort(); // Reserve a genuinely cold bank.
    if(warm){
        at::reset_status(p);microbatch();microbatch();configure();
        require(adam.step(options(1)).committed,"direct owner warmup did not commit");
    }
    const auto moments=snapshot_states();
    require(moments.size()==p.size(),"owner baseline registry");
    bool nonzero_moment=false;
    for(const auto& s:moments){
        require(s.initialized==warm,"owner fixture is not cold/warm as requested");
        if(warm){finite(s.m);finite(s.v);for(int i=0;i<s.m.size;++i)nonzero_moment|=s.m.data()[i]!=0;}
        else require(!s.m.size&&!s.v.size,"cold bank exported initialized moments");
    }
    require(!warm||nonzero_moment,"warm owner must have real numerical moment state");
    std::vector<Tensor> weights,gradients;
    std::vector<uint64_t> versions;
    std::vector<const float*> storage,gradient_storage;
    for(auto* a:p){weights.push_back(a->data.cpu().clone());versions.push_back(a->version);
        storage.push_back(a->data.raw_data());gradient_storage.push_back(a->grad.raw_data());}
    const auto clean_bias=q.bias.data.clone();
    const auto bad_bias=upload(q.bias.data.shape.dims,std::vector<float>(q.bias.data.size,1e4f));
    zero_gradients();at::reset_status(p);
    const auto dispatch_before=at::dispatch_counters();
    // Controlled exact FP32 leaf injection preserves storage AND version in
    // an established owner. Never mutate a projection between its F and B.
    q.bias.data.copy_from(bad_bias);microbatch();
    require(issue()!=0,"finite invalid Q bias did not produce real Attention status");
    q.bias.data.copy_from(clean_bias);microbatch();
    require(issue()!=0,"valid following microbatch erased sticky Attention status");
    const auto dispatch_ready=at::dispatch_counters();
    require(dispatch_ready[0]>=dispatch_before[0]+2&&dispatch_ready[1]>=dispatch_before[1]+2,
            "direct owner fault group did not complete both real forward/backward calls");
    bool nonzero_gradient=false;
    for(size_t i=0;i<p.size();++i){
        auto* a=p[i];require(a->has_gradient(),"dense projection contribution missing");
        finite(a->data);finite(a->grad);gradients.push_back(a->grad.cpu().clone());
        for(int j=0;j<gradients.back().size;++j)nonzero_gradient|=gradients.back().data()[j]!=0;
        require(equal(a->data,weights[i])&&a->version==versions[i]&&a->data.raw_data()==storage[i]&&
                a->grad.raw_data()==gradient_storage[i],"fault injection altered owner identity or restored master");
    }
    require(nonzero_gradient,"finite valid microbatch must supply a real dense gradient");
    configure();
    const auto merge_before=at::dispatch_counters()[2];
    // No exception counts as success: the numeric gate must reject recoverably.
    require(!adam.step(options(warm?2:1)).committed,"owner without callback ignored sticky Attention status");
    require(at::dispatch_counters()[2]>merge_before,"GpuSparseAdam::step did not merge Attention itself");
    for(size_t i=0;i<p.size();++i){
        require(equal(p[i]->data,weights[i])&&p[i]->version==versions[i]&&p[i]->data.raw_data()==storage[i],
                "rejected direct owner changed weight/storage/version");
        require(equal(p[i]->grad,gradients[i])&&p[i]->grad.raw_data()==gradient_storage[i],
                "rejected direct owner scaled or replaced gradients before the gate");
    }
    same_states(moments);
    require(issue()!=0,"owner abort unexpectedly cleared provider's sticky ledger");

    zero_gradients();at::reset_status(p);require(issue()==0,"explicit group reset did not clear status");
    microbatch();microbatch();require(issue()==0,"clean recovery produced Attention status");
    configure();const auto recovered=adam.step(options(warm?2:1));
    require(recovered.committed&&std::isfinite(recovered.norm)&&recovered.norm>0,
            "owner must recover on clean group after reset_status");
    const auto recovered_states=adam.snapshot();bool weight_changed=false;
    for(size_t i=0;i<p.size();++i){
        require(p[i]->version==versions[i]+1&&p[i]->data.raw_data()==storage[i],"recovered owner version/storage publication");
        require(recovered_states[i].initialized&&recovered_states[i].version==p[i]->version,
                "recovered owner did not publish moments/version");
        finite(recovered_states[i].m);finite(recovered_states[i].v);
        weight_changed|=!equal(p[i]->data,weights[i]);
    }
    require(weight_changed,"recovered direct owner must perform a real Adam update");
    std::cout<<"PASS direct Attention owner without callback "<<(warm?"warm":"cold")
             <<(deterministic?" deterministic":" ordinary")<<" sticky finite gate and recovery\n";
}

void composition_and_ledger(){
    set_policy("rdna_bf16_v1");
    BitLinear q(8,8,true,91),kv(8,8,true,92),o(8,8,true,93);
    q.set_exact_linear_mode(true);kv.set_exact_linear_mode(true);o.set_exact_linear_mode(true);
    q.to(Device::GPU);kv.to(Device::GPU);o.to(Device::GPU);
    auto p=params(q,kv,o);at::Provider provider(q,kv,o);
    const at::Config c{{2,3,2,1,4,3,0.5f,ar::Precision::BF16},0};
    auto x=upload({2,3,8},std::vector<float>(48,0.125f));
    auto cs=upload({3,2},std::vector<float>(6,1)),sn=upload({3,2},std::vector<float>(6,0));
    auto dy=upload({2,3,8},std::vector<float>(48,0.0625f));
    std::vector<Tensor> first;
    for(int micro=0;micro<2;++micro){
        (void)provider.forward(x,c,cs,sn,{3,1});rejects([&]{provider.forward(x,c,cs,sn,{3,1});});
        Tensor dx=provider.backward(dy);require(dx.shape==x.shape,"input VJP geometry");
        if(micro==0) for(auto* a:p) first.push_back(a->grad.size?a->grad.cpu().clone():Tensor());
        else for(std::size_t j=0;j<p.size();++j) if(first[j].size){
            auto g=p[j]->grad.cpu();for(std::size_t i=0;i<g.size;++i) near(g.data()[i],2*first[j].data()[i],"projection accumulation");
        }
    }
    cuda_detail::DeviceBuffer<int> flag;require(flag.ensure(1),"finite flag allocation");
    auto issue=[&]{
        require(cudaMemsetAsync(flag.get(),0,sizeof(int),gpu::current_stream())==cudaSuccess,"clear trainer flag");
        at::merge_status(p,flag.get());require(cudaStreamSynchronize(gpu::current_stream())==cudaSuccess,"finite fence");
        int v=-1;require(cudaMemcpy(&v,flag.get(),sizeof(v),cudaMemcpyDeviceToHost)==cudaSuccess,"finite download");return v;
    };
    require(issue()==0,"valid projection ledger");
    (void)provider.forward(x,c,cs,sn,{3,1});
    auto bad=upload({2,3,8},std::vector<float>(48,INFINITY));(void)provider.backward(bad);
    require(issue()!=0,"invalid dO reaches trainer gate");
    (void)provider.forward(x,c,cs,sn,{3,1});(void)provider.backward(dy);
    require(issue()!=0,"failure must persist across microbatch replacement");
    at::reset_status(p);require(issue()==0,"group reset clears ledger");
    // Owning model context and trainer context are different execution lanes.
    gpu::ExecutionContext lane;at::Provider contextual(q,kv,o);
    {gpu::ExecutionContext::Scope scope(lane);(void)contextual.forward(x,c,cs,sn,{3,1});(void)contextual.backward(dy);}
    require(issue()==0,"event bridge to trainer lane");at::reset_status(p);
    {gpu::ExecutionContext::Scope scope(lane);(void)contextual.forward(x,c,cs,sn,{3,1});(void)contextual.backward(dy);}
    require(issue()==0,"event bridge from group reset");
    (void)provider.forward(x,c,cs,sn,{3,1});p.front()->mark_updated();
    rejects([&]{provider.backward(dy);});provider.cancel_pending();
    set_policy("fp32");rejects([&]{provider.forward(x,c,cs,sn,{3,1});});
}
}
int main(){
    try{
        require(at::parse_policy("")==at::Policy::FP32,"default FP32");
        rejects([]{at::parse_policy("bf16");});rejects([]{at::parse_policy("rdna_bf16_v2");});
        require(gpu::select_preferred_device(),"registered GPU test requires hardware");
        require(ar::supported({2,5,4,2,3,3,0.5f,ar::Precision::BF16}),"registered GPU test requires compiled RDNA3 provider");
        analytic_tape(ar::Precision::BF16);analytic_tape(ar::Precision::FP16);composition_and_ledger();
        for(bool deterministic:{true,false}) for(bool warm:{false,true})
            direct_sparse_owner_status(warm,deterministic);
        std::cout<<"PASS attention owned tape/projections/accumulation/finite gate\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
