// Real Jamba/Trainer/checkpoint consumer. Run only after root applies integration.
#include "gpu_attention_training.h"
#include "jamba.h"
#include "trainer.h"
#include "nsos/determinism.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cstring>
using namespace nsos;
namespace at=nsos::attention_training;
namespace {
void require(bool v,const char* msg){if(!v) throw std::runtime_error(msg);}
void env(const char* k,const char* v){
#ifdef _WIN32
    require(_putenv_s(k,v)==0,"environment mutation");
#else
    require(setenv(k,v,1)==0,"environment mutation");
#endif
}
ModelConfig config(){
    ModelConfig c;c.num_layers=1;c.d_model=16;c.vocab_size=32;c.n_heads=2;c.n_kv_heads=1;
    c.mamba_head_dim=8; // Validation applies Mamba geometry even in attention-only schedules.
    c.attention_period=1;c.attention_slot=0;c.force_mamba_last_layer=false;
    c.hybrid_composition=HybridComposition::LegacyReplacement;
    c.faithful_attention_linears=true;c.use_moe=false;c.use_ttt=false;c.use_kan=false;c.use_chrass=false;
    c.dropout=0;c.tie_word_embeddings=true;c.use_gradient_checkpointing=false;
    return c;
}
void setup(Trainer& t,int accumulation){
    TrainPhaseScheduler s;s.progressive_qat_enabled=false;t.configure_progressive_qat(s);
    t.loss_scale=1;t.weight_decay=0;t.warmup_steps=1;t.total_training_steps=16;
    t.gradient_accumulation_steps=accumulation;t.max_grad_norm=1;
}
std::vector<Tensor> snapshot(JambaModel& m){std::vector<Tensor> r;for(auto* p:m.parameters())r.push_back(p->data.cpu().clone());return r;}
void same(JambaModel& m,const std::vector<Tensor>& s){
    auto p=m.parameters();require(p.size()==s.size(),"registry size");
    for(std::size_t i=0;i<p.size();++i){auto v=p[i]->data.cpu();require(v.shape==s[i].shape,"parameter shape");
        require(std::memcmp(v.data(),s[i].data(),v.size*sizeof(float))==0,"parameter mismatch");}
}
float group(Trainer& t){
    (void)t.accumulate_microbatch({1,3,5,7},{3,5,7,9});
    (void)t.accumulate_microbatch({2,4,6},{4,6,8});
    return t.commit_optimizer_step(2);
}
void continuation(const char* mode){
    env("NSOS_ATTN_TRAINING_PROVIDER",mode);
    JambaModel m(config(),Device::GPU);m.set_reference_path(true);m.set_training_mode(true);
    require(m.layers.front()->attn_layer!=nullptr,"attention fixture must be active");
    Trainer t(&m,1e-3f);setup(t,2);
    const auto before=at::dispatch_counters();
    (void)t.accumulate_microbatch({1,3,5,7},{3,5,7,9});
    bool q=false,kv=false,out=false;
    for(auto* p:m.parameters())if(p->grad.size&&p->grad.norm()>0){
        q|=p->name.find(".attn.q_down_proj.weight")!=std::string::npos;
        kv|=p->name.find(".attn.kv_down_proj.weight")!=std::string::npos;
        out|=p->name.find(".attn.out_proj.weight")!=std::string::npos;
    }
    require(q&&kv&&out,"real model must reach all attention projection gradients");
    (void)t.accumulate_microbatch({2,4,6},{4,6,8});
    require(std::isfinite(t.commit_optimizer_step(2)),"first accumulated step");
    auto count=at::dispatch_counters();require(count[0]>=before[0]+2&&count[1]>=before[1]+2&&count[2]>before[2],"real consumer dispatch/gate evidence");
    bool identified=false;
    for(const auto& f:t.execution_identity_fields())if(f.first=="attention.training_provider") identified=f.second==mode;
    require(identified,"checkpoint identity must describe active provider");
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto stem=std::filesystem::temp_directory_path()/ ("nsos_attention_consumer_"+std::to_string(stamp));
    const auto model_file=stem.string()+".model",state_file=stem.string()+".state";
    t.capture_checkpoint_snapshot()->write(model_file,state_file);
    JambaModel resumed(config(),Device::GPU);resumed.load(model_file);
    Trainer r(&resumed,9.0f);setup(r,2);
    r.load_training_state(state_file,model_file);
    require(r.execution_identity_digest()==t.execution_identity_digest(),"resume policy fingerprint");
    const float a=group(t),b=group(r);
    require(a==b,"resumed accumulated loss must agree exactly");same(resumed,snapshot(m));
    const auto mp=m.parameters(),rp=resumed.parameters();
    for(std::size_t i=0;i<mp.size();++i)for(int moment=0;moment<2;++moment){
        const auto& ma=moment?t.v_state:t.m_state;const auto& mb=moment?r.v_state:r.m_state;
        auto ia=ma.find(mp[i]),ib=mb.find(rp[i]);require((ia==ma.end())==(ib==mb.end()),"resumed optimizer cohort");
        if(ia!=ma.end()){auto x=ia->second.cpu(),y=ib->second.cpu();require(x.shape==y.shape&&std::memcmp(x.data(),y.data(),x.size*sizeof(float))==0,"resumed Adam moments");}
    }
    env("NSOS_ATTN_TRAINING_PROVIDER",std::string(mode)=="rdna_bf16_v1"?"rdna_fp16_v1":"rdna_bf16_v1");
    JambaModel wrong(config(),Device::GPU);wrong.load(model_file);
    Trainer w(&wrong);setup(w,2);const auto original=snapshot(wrong);
    bool rejected=false;try{w.load_training_state(state_file,model_file);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"different attention precision must reject resume");same(wrong,original);
    env("NSOS_ATTN_TRAINING_PROVIDER",mode);
    std::filesystem::remove(model_file);std::filesystem::remove(state_file);
}
void numeric_rejection(){
    env("NSOS_ATTN_TRAINING_PROVIDER","rdna_bf16_v1");
    JambaModel m(config(),Device::GPU);m.set_reference_path(true);m.set_training_mode(true);
    Parameter* bias=nullptr;
    for(auto* p:m.parameters())if(p->name.find(".attn.q_down_proj.bias")!=std::string::npos)bias=p;
    require(bias!=nullptr,"Q projection bias fixture");const Tensor saved=bias->data.clone();
    bias->copy_data_from(Tensor::ones(bias->data.shape,Device::GPU).mul(1e4f));
    Trainer t(&m,1e-3f);setup(t,1);const auto before=snapshot(m);
    bool rejected=false;try{(void)t.train_step({1,3,5,7},{3,5,7,9});}catch(const std::runtime_error&){rejected=true;}
    require(rejected&&t.last_optimizer_step_skipped&&t.global_step_count==0,"device status must reject optimizer commit");same(m,before);
    // Existing Trainer preflight may materialize zero moments before rejection.
    // Require their numerical state to remain uncommitted, rather than emptiness.
    for(const auto* states:{&t.m_state,&t.v_state})for(const auto& entry:*states){
        auto h=entry.second.cpu();for(int i=0;i<h.size;++i)require(h.data()[i]==0,"cold rejection changed Adam moments");
    }
    bias->copy_data_from(saved);require(std::isfinite(t.train_step({1,3,5,7},{3,5,7,9})),"clean group must recover after rejection");
    std::unordered_map<Parameter*,Tensor> old_m,old_v;
    for(const auto& entry:t.m_state)old_m.emplace(entry.first,entry.second.cpu().clone());
    for(const auto& entry:t.v_state)old_v.emplace(entry.first,entry.second.cpu().clone());
    bias->copy_data_from(Tensor::ones(bias->data.shape,Device::GPU).mul(1e4f));
    const auto warm_before=snapshot(m);rejected=false;
    try{(void)t.train_step({1,3,5,7},{3,5,7,9});}catch(const std::runtime_error&){rejected=true;}
    require(rejected&&t.last_optimizer_step_skipped&&t.global_step_count==1,"warm status rejection must preserve committed step");same(m,warm_before);
    auto same_moments=[](const auto& a,const auto& b){require(a.size()==b.size(),"rejected optimizer cohort changed");
        for(const auto& entry:a){auto found=b.find(entry.first);require(found!=b.end(),"rejected moment missing");
            auto h=found->second.cpu();require(h.shape==entry.second.shape&&std::memcmp(h.data(),entry.second.data(),h.size*sizeof(float))==0,"warm rejection changed Adam moments");}
    };
    same_moments(old_m,t.m_state);same_moments(old_v,t.v_state);
    bias->copy_data_from(saved);require(std::isfinite(t.train_step({1,3,5,7},{3,5,7,9})),"warm group must recover after rejection");
}
}
int main(){
    try{
        env("NSOS_ATTN_TILED_TRAINING","0");env("NSOS_ATTN_BWD_HOST","0");
        set_matmul_precision_mode(0);set_strict_gpu_execution(true);determinism::set_deterministic_reductions(true);
        require(gpu::select_preferred_device(),"registered GPU consumer test requires hardware");
        require(attention_rdna::supported({1,4,2,1,8,4,0.353553f,attention_rdna::Precision::BF16}),"registered GPU consumer requires RDNA3 binary coverage");
        env("NSOS_HEAD_CCE","0");continuation("rdna_bf16_v1");
        env("NSOS_HEAD_CCE","1");continuation("rdna_bf16_v1");continuation("rdna_fp16_v1");numeric_rejection();
        std::cout<<"PASS real Attention/Jamba/Trainer/accumulation/checkpoint/status consumer\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
