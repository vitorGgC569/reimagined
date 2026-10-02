#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "gpu_parity_common.h"
#include "trainer.h"
#include "muon_math.h"
#include "nsos/determinism.h"
#include <chrono>
#include <filesystem>
#include <cstring>

using namespace nsos;
namespace {
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Policy {
    const char* name;std::string previous;bool existed;
    Policy(const char* n,const char* value):name(n),existed(std::getenv(n)!=nullptr){if(existed)previous=std::getenv(n);set(value);}
    void set(const char* value){
#ifdef _WIN32
        require(_putenv_s(name,value)==0,"policy write failed");
#else
        require(setenv(name,value,1)==0,"policy write failed");
#endif
    }
    ~Policy(){
#ifdef _WIN32
        _putenv_s(name,existed?previous.c_str():"");
#else
        if(existed)setenv(name,previous.c_str(),1);else unsetenv(name);
#endif
    }
};
ModelConfig config(){ModelConfig c;c.architecture_schema_version=3;c.num_layers=1;c.d_model=16;c.vocab_size=37;
    c.n_heads=2;c.n_kv_heads=1;c.attention_period=64;c.force_mamba_last_layer=true;c.mamba3_enabled=true;c.mamba3_schema_version=1;
    c.mamba3_state_dim=128;c.mamba3_mimo=true;c.mamba3_mimo_rank=2;c.mamba3_outproj_norm=true;c.mamba_expand=1;c.mamba_head_dim=8;c.mamba_n_groups=1;
    c.use_moe=false;c.use_ttt=false;c.use_kan=false;c.use_chrass=false;c.dropout=0;c.use_gradient_checkpointing=false;c.tie_word_embeddings=true;c.max_context_tokens=32;return c;}
void prepare(JambaModel& model){model.set_reference_path(true);model.set_training_mode(true);require(model.layers[0]->mamba3_layer!=nullptr,"Mamba3 absent");}
void setup(Trainer& t){t.phase_scheduler.progressive_qat_enabled=false;t.phase_scheduler.ternary_regularization=0;t.moe_aux_loss_scale=0;t.logit_l2_beta=0;t.pantheon_vib_beta=0;t.repetition_unlikelihood_scale=0;
    t.dynamic_loss_scaling_enabled=false;t.optimizer_state_bits=32;t.gradient_accumulation_steps=2;t.scheduler_unit=Trainer::SchedulerUnit::Tokens;t.training_tokens=1000;t.warmup_tokens=0;t.decay_tokens=0;
    t.weight_decay=.1f;t.max_grad_norm=1;t.warmup_steps=0;t.total_training_steps=100;t.min_learning_rate_scale=1;}
void micros(Trainer& t){t.accumulate_microbatch({1,2,3,4,5,6,7,8},{2,3,4,5,6,7,8,9});t.accumulate_microbatch({9,8,7,6,5},{8,7,6,5,4});}
bool equal(const Tensor& a,const Tensor& b){auto x=a.cpu(),y=b.cpu();return x.shape==y.shape && !std::memcmp(x.data(),y.data(),size_t(x.size)*sizeof(float));}
std::vector<GpuSparseAdamState> owner_states(Trainer& t){gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());return t.device_sparse_adam->snapshot();}
void run(){
    Policy optimizer("NSOS_OPTIMIZER","muon_ns5_fp32_v1"),muon_lr("NSOS_MUON_LR","0.02"),epilogue("NSOS_OPTIMIZER_FUSED_EPILOGUE","1"),moe("NSOS_MOE_DEVICE_ADAM","0"),det("NSOS_DETERMINISTIC","1");
    set_matmul_precision_mode(0);JambaModel model(config(),Device::GPU);prepare(model);Trainer trainer(&model,.002f);setup(trainer);
    // A dense Mamba3 topology must open a device transaction without inventing
    // a MoE producer. A zero contribution still initializes lazy state/versions.
    micros(trainer);require(trainer.device_sparse_group_open && trainer.device_moe_groups.empty(),"dense Muon did not open its transaction");
    auto no_decay=model.no_weight_decay_parameters(); // authoritative Mamba3 dt_bias/D registry
    std::vector<Tensor> protected_weights;for(auto* p:no_decay)protected_weights.push_back(p->data.cpu().clone());
    const auto parameters=model.parameters();std::vector<uint64_t> versions;for(auto* p:parameters){versions.push_back(p->version);if(p->trainable && p->grad.size)p->grad.copy_from(Tensor::zeros(p->grad.shape,Device::GPU));}
    require(std::isfinite(trainer.commit_optimizer_step(2)),"zero-gradient commit nonfinite");
    for(size_t i=0;i<no_decay.size();++i)require(equal(no_decay[i]->data,protected_weights[i]),"noDecay dt_bias/D changed under zero gradient");
    const auto states=owner_states(trainer);size_t matrices=0;
    for(const auto& state:states){const auto found=std::find_if(parameters.begin(),parameters.end(),[&](Parameter* p){return p->name==state.name;});require(found!=parameters.end(),"snapshot name absent");const bool muon=muon::hidden_matrix((*found)->name,(*found)->data.shape.dims);matrices+=muon;
        require(state.algorithm==(muon?GpuSparseAlgorithm::MuonNs5Fp32:GpuSparseAlgorithm::AdamW),"Muon partition differs");if(muon && state.initialized){const auto v=state.v.cpu();for(int i=0;i<v.size;++i)require(v.data()[i]==0,"Muon V is not exactly zero");}}
    require(matrices>=2,"hidden Mamba3 matrices not selected");require(trainer.tokens_committed==13 && trainer.tokens_processed==13,"token commit history differs");
    auto fields=trainer.execution_identity_fields();require(std::find(fields.begin(),fields.end(),RuntimeExecutionIdentity::Field{"optimizer.algorithm",muon::kIdentity})!=fields.end(),"Muon checkpoint identity absent");
    const std::string stem=(std::filesystem::temp_directory_path()/("nsos_muon_dense_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    struct Files{std::string model,state;~Files(){std::error_code ec;std::filesystem::remove(model,ec);std::filesystem::remove(state,ec);}}files{stem+".model",stem+".state"};
    trainer.capture_checkpoint_snapshot()->write(files.model,files.state);
    JambaModel resumed(config(),Device::GPU);prepare(resumed);resumed.load(files.model);Trainer rt(&resumed,.002f);setup(rt);rt.load_training_state(files.state,files.model);
    require(rt.tokens_processed==trainer.tokens_processed && rt.tokens_committed==trainer.tokens_committed && rt.execution_identity_digest()==trainer.execution_identity_digest(),"portable resume metadata differs");
    micros(trainer);micros(rt);const float a=trainer.commit_optimizer_step(2),b=rt.commit_optimizer_step(2);require(a==b && std::isfinite(a),"portable resume loss differs");
    const auto restored=resumed.parameters();require(parameters.size()==restored.size(),"resumed registry differs");
    for(size_t i=0;i<parameters.size();++i)require(parameters[i]->name==restored[i]->name && equal(parameters[i]->data,restored[i]->data),"portable resume weights differ");
    const auto left=owner_states(trainer),right=owner_states(rt);require(left.size()==right.size(),"portable resume state count differs");
    for(size_t i=0;i<left.size();++i){require(left[i].name==right[i].name && left[i].algorithm==right[i].algorithm && left[i].initialized==right[i].initialized,"portable resume kind/lazy state differs");if(left[i].initialized)require(equal(left[i].m,right[i].m)&&equal(left[i].v,right[i].v),"portable resume moments differ");}
    require(trainer.global_step_count==2 && rt.global_step_count==2 && trainer.tokens_committed==26 && rt.tokens_committed==26,"portable trajectory counters differ");
    // Switching policy must fail before sidecar publication, even with explicit
    // legacy migration permission. CPU checks remain unsupported, fail-closed.
    optimizer.set("adamw");bool rejected=false;try{Trainer wrong(&resumed,.002f);setup(wrong);wrong.load_training_state(files.state,files.model,true);}catch(const std::exception&){rejected=true;}require(rejected,"Muon sidecar loaded as Adam");optimizer.set("muon_ns5_fp32_v1");
    // Explicit abort of a pending dense group resets all group metadata and
    // leaves commit counters/weights intact; the next valid group can recover.
    trainer.accumulate_microbatch({1,2,3},{2,3,4});trainer.abort_gradient_accumulation();require(!trainer.device_sparse_group_open && trainer.pending_accumulation_microbatches==0 && trainer.pending_accumulated_tokens==0 && trainer.tokens_committed==26,"dense abort did not reset group");
    micros(trainer);require(std::isfinite(trainer.commit_optimizer_step(2)),"dense abort recovery failed");
}
}
int main()try{gpu_parity_test::require_cuda_device("muon_trainer");run();std::cout<<"Muon Trainer dense Mamba3 tied-head A2/tokens: noDecay dt_bias/D, partition/identity, portable checkpoint/continuation, policy rejection and abort/recovery passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
