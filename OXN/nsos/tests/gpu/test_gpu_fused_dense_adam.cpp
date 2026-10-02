#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "gpu_parity_common.h"
#include "trainer.h"
#include "gpu_execution.h"
#include "optimizer_runtime_policy.h"
#include <chrono>
#include <filesystem>
#include <cstring>
#include <map>

using namespace nsos;
namespace {
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Policy {
    const char* name;std::string prior;bool existed;
    Policy(const char* n,const char* value):name(n),existed(std::getenv(n)!=nullptr){if(existed)prior=std::getenv(n);set(value);}
    void set(const char* value){
#ifdef _WIN32
        require(_putenv_s(name,value)==0,"policy write failed");
#else
        require(setenv(name,value,1)==0,"policy write failed");
#endif
    }
    ~Policy(){
#ifdef _WIN32
        _putenv_s(name,existed?prior.c_str():"");
#else
        if(existed)setenv(name,prior.c_str(),1);else unsetenv(name);
#endif
    }
};
template<class F>void rejects(F&& f,const char* message){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,message);}
ModelConfig config(){ModelConfig c;c.architecture_schema_version=3;c.num_layers=1;c.d_model=16;c.vocab_size=37;c.n_heads=2;c.n_kv_heads=1;c.attention_period=64;c.force_mamba_last_layer=true;
    c.mamba3_enabled=true;c.mamba3_schema_version=1;c.mamba3_state_dim=128;c.mamba3_mimo=true;c.mamba3_mimo_rank=2;c.mamba3_outproj_norm=true;c.mamba_expand=1;c.mamba_head_dim=8;c.mamba_n_groups=1;
    c.use_moe=false;c.use_ttt=false;c.use_kan=false;c.use_chrass=false;c.dropout=0;c.use_gradient_checkpointing=false;c.tie_word_embeddings=true;c.max_context_tokens=32;return c;}
void prepare(JambaModel& m){m.set_reference_path(true);m.set_training_mode(true);require(m.layers[0]->mamba3_layer!=nullptr && !m.layers[0]->uses_moe(),"dense Mamba3 fixture missing");}
void setup(Trainer& t){t.phase_scheduler.progressive_qat_enabled=false;t.phase_scheduler.ternary_regularization=0;t.moe_aux_loss_scale=0;t.logit_l2_beta=0;t.pantheon_vib_beta=0;t.repetition_unlikelihood_scale=0;t.dynamic_loss_scaling_enabled=false;
    t.optimizer_state_bits=32;t.gradient_accumulation_steps=2;t.scheduler_unit=Trainer::SchedulerUnit::Tokens;t.training_tokens=1000;t.warmup_tokens=0;t.decay_tokens=0;t.weight_decay=.1f;t.max_grad_norm=.01f;t.warmup_steps=0;t.total_training_steps=100;t.min_learning_rate_scale=1;}
void micros(Trainer& t){t.accumulate_microbatch({1,2,3,4,5,6,7,8},{2,3,4,5,6,7,8,9});t.accumulate_microbatch({9,8,7,6,5},{8,7,6,5,4});}
bool exact(const Tensor& a,const Tensor& b){auto x=a.cpu(),y=b.cpu();return x.shape==y.shape && !std::memcmp(x.data(),y.data(),size_t(x.size)*sizeof(float));}
void near(const Tensor& a,const Tensor& b,float tolerance,const char* text){auto x=a.cpu(),y=b.cpu();require(x.shape==y.shape,text);for(int i=0;i<x.size;++i)require(std::isfinite(x.data()[i])&&std::isfinite(y.data()[i])&&std::abs(x.data()[i]-y.data()[i])<=tolerance,text);}
void models(JambaModel& a,JambaModel& b,float tolerance){const auto x=a.parameters(),y=b.parameters();require(x.size()==y.size(),"registry count differs");for(size_t i=0;i<x.size();++i){require(x[i]->name==y[i]->name && x[i]->trainable==y[i]->trainable,"registry identity differs");near(x[i]->data,y[i]->data,tolerance,"dense fused/reference weight trajectory differs");}}
std::vector<GpuSparseAdamState> owner_states(Trainer& t){gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());return t.device_sparse_adam->snapshot();}
void moments(Trainer& ordinary,Trainer& fused,float tolerance){
    const auto states=owner_states(fused);std::map<std::string,Parameter*> reference;for(auto* p:ordinary.model->parameters())reference[p->name]=p;
    for(const auto& state:states){require(state.algorithm==GpuSparseAlgorithm::AdamW,"dense Adam selected Muon");auto* p=reference.at(state.name);const auto m=ordinary.m_state.find(p),v=ordinary.v_state.find(p);require((m!=ordinary.m_state.end())==(v!=ordinary.v_state.end()),"reference moments incomplete");require(state.initialized==(m!=ordinary.m_state.end()),"dense lazy presence differs");if(state.initialized){near(state.m,m->second,tolerance,"dense first moment differs");near(state.v,v->second,tolerance,"dense second moment differs");}}
}
void counters(const Trainer& a,const Trainer& b,int steps){require(a.global_step_count==steps && b.global_step_count==steps && a.tokens_processed==steps*13 && b.tokens_processed==steps*13 && a.tokens_committed==steps*13 && b.tokens_committed==steps*13,"dense token/scheduler counters differ");}
void run(bool deterministic){
    Policy algorithm("NSOS_OPTIMIZER","adamw"),epilogue("NSOS_OPTIMIZER_FUSED_EPILOGUE","0"),legacy("NSOS_MOE_DEVICE_ADAM","0"),det("NSOS_DETERMINISTIC",deterministic?"1":"0"),clip("NSOS_DEVICE_GRAD_CLIP","1"),cce("NSOS_HEAD_CCE","0");
    set_matmul_precision_mode(0);JambaModel ordinary_model(config(),Device::GPU),fused_model(config(),Device::GPU);prepare(ordinary_model);prepare(fused_model);
    const auto op=ordinary_model.parameters(),fp=fused_model.parameters();require(op.size()==fp.size(),"initial registry mismatch");for(size_t i=0;i<op.size();++i){require(op[i]->name==fp[i]->name,"initial names mismatch");fp[i]->data.copy_from(op[i]->data);fp[i]->trainable=op[i]->trainable;}
    Trainer ordinary(&ordinary_model,.002f),fused(&fused_model,.002f);setup(ordinary);setup(fused);
    epilogue.set("0");ordinary.validate_execution_identity();epilogue.set("1");fused.validate_execution_identity();
    const auto fields=fused.execution_identity_fields();require(std::find(fields.begin(),fields.end(),RuntimeExecutionIdentity::Field{"optimizer.dense_gradient_bank","explicit_dense_contribution_finite_lazy_snapshot_versions_v1"})!=fields.end(),"dense bank identity absent");
    // Explicit legacy opt-in still refuses a dense topology, even if the fused
    // option is also enabled; CPU/quantized moment guard rejects before mutation.
    legacy.set("1");rejects([&]{Trainer bad(&fused_model,.002f);setup(bad);bad.validate_execution_identity();},"legacy MoE flag accepted dense topology");legacy.set("0");
    rejects([&]{Trainer bad(&fused_model,.002f);setup(bad);bad.optimizer_state_bits=4;bad.validate_execution_identity();},"fused Adam accepted 4-bit state");
    {JambaModel cpu(config(),Device::CPU);Trainer bad(&cpu,.002f);setup(bad);rejects([&]{bad.validate_execution_identity();},"fused Adam silently fell back to CPU");}
    // Zero is a real contribution. Both policies execute identical explicit
    // decay while authoritative Mamba3 dt_bias/D remain bitwise unchanged.
    const auto nd=fused_model.no_weight_decay_parameters();require(nd.size()==2,"Mamba3 no-decay registry differs");std::vector<Tensor> protected_weights;for(auto* p:nd)protected_weights.push_back(p->data.cpu().clone());
    epilogue.set("0");micros(ordinary);for(auto* p:op)if(p->trainable && p->grad.size)p->grad.copy_from(Tensor::zeros(p->grad.shape,Device::GPU));const float z0=ordinary.commit_optimizer_step(2);
    epilogue.set("1");micros(fused);require(fused.device_sparse_group_open && fused.device_moe_groups.empty(),"fused dense Adam invented a MoE producer");for(auto* p:fp)if(p->trainable && p->grad.size)p->grad.copy_from(Tensor::zeros(p->grad.shape,Device::GPU));const float z1=fused.commit_optimizer_step(2);
    require(std::isfinite(z0)&&std::isfinite(z1)&&std::abs(z0-z1)<2e-6f,"zero contribution loss differs");for(size_t i=0;i<nd.size();++i)require(exact(nd[i]->data,protected_weights[i]),"zero gradient decayed dt_bias/D");models(ordinary_model,fused_model,2e-6f);moments(ordinary,fused,2e-6f);counters(ordinary,fused,1);
    // The next group consumes the clearing proof and omits the full-bank zero.
    // Compare to an independent legacy Trainer with explicit clipping tolerance.
    epilogue.set("0");micros(ordinary);const float a=ordinary.commit_optimizer_step(2);
    epilogue.set("1");micros(fused);const float b=fused.commit_optimizer_step(2);require(std::abs(a-b)<2e-5f && std::isfinite(b),"clipped dense loss differs");require(ordinary.last_update_was_clipped && fused.last_update_was_clipped,"fixture did not exercise global clipping");models(ordinary_model,fused_model,5e-6f);moments(ordinary,fused,5e-6f);counters(ordinary,fused,2);
    // One-use proof is inspectable independently of the Trainer elision.
    {gpu::ExecutionContext::Scope lane(fused.device_sparse_execution_context());require(fused.device_sparse_adam->consume_fused_gradient_reset(fp),"committed dense clearing proof absent");require(!fused.device_sparse_adam->consume_fused_gradient_reset(fp),"dense clearing proof reusable");}
    const std::string stem=(std::filesystem::temp_directory_path()/("nsos_fused_dense_adam_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();struct Files{std::string model,state;~Files(){std::error_code ec;std::filesystem::remove(model,ec);std::filesystem::remove(state,ec);}}files{stem+".model",stem+".state"};
    fused.capture_checkpoint_snapshot()->write(files.model,files.state);JambaModel restored(config(),Device::GPU);prepare(restored);restored.load(files.model);Trainer resumed(&restored,.002f);setup(resumed);resumed.load_training_state(files.state,files.model);require(!resumed.device_sparse_adam,"cold resume materialized owner before group");require(resumed.execution_identity_digest()==fused.execution_identity_digest(),"cold resume policy differs");counters(fused,resumed,2);
    epilogue.set("0");rejects([&]{Trainer wrong(&restored,.002f);setup(wrong);wrong.load_training_state(files.state,files.model,true);},"fused sidecar resumed under ordinary Adam");micros(ordinary);const float c=ordinary.commit_optimizer_step(2);
    epilogue.set("1");micros(fused);const float d=fused.commit_optimizer_step(2);micros(resumed);const float e=resumed.commit_optimizer_step(2);const float continuation_tolerance=deterministic?0.0f:5e-6f;require(std::abs(d-e)<=continuation_tolerance && std::abs(c-d)<3e-5f,"cold resume or third-step reference loss differs");models(fused_model,restored,continuation_tolerance);models(ordinary_model,fused_model,7e-6f);moments(ordinary,fused,7e-6f);counters(ordinary,fused,3);counters(fused,resumed,3);
    const auto fs=owner_states(fused),rs=owner_states(resumed);require(fs.size()==rs.size(),"cold resumed state count differs");for(size_t i=0;i<fs.size();++i){require(fs[i].name==rs[i].name && fs[i].initialized==rs[i].initialized && fs[i].algorithm==rs[i].algorithm,"cold resume state identity differs");if(fs[i].initialized){near(fs[i].m,rs[i].m,continuation_tolerance,"cold resumed first moment differs");near(fs[i].v,rs[i].v,continuation_tolerance,"cold resumed second moment differs");}}
    const auto committed=fused.tokens_committed;std::vector<Tensor> before;for(auto* p:fp)before.push_back(p->data.cpu().clone());fused.accumulate_microbatch({1,2,3},{2,3,4});fused.abort_gradient_accumulation();require(!fused.device_sparse_group_open && fused.pending_accumulation_microbatches==0 && fused.pending_accumulated_tokens==0 && fused.tokens_committed==committed,"dense abort metadata differs");for(size_t i=0;i<fp.size();++i)require(exact(fp[i]->data,before[i]),"dense abort modified weight");micros(fused);require(std::isfinite(fused.commit_optimizer_step(2)),"dense abort recovery failed");
}
}
int main()try{gpu_parity_test::require_cuda_device("fused_dense_adam");run(true);run(false);std::cout<<"Fused dense Adam: legacy noMoE reject, CPU/state guards, Mamba3 tied-head/A2/tokens, noDecay, clipped legacy trajectory, reset proof, cold resume, policy rejection and abort/recovery passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
