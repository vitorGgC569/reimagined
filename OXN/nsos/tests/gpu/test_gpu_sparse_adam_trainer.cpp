#include "gpu_parity_common.h"
#include "jamba.h"
#include "trainer.h"
#include "nsos/determinism.h"
#include <array>
#include <filesystem>
#include <chrono>

using namespace nsos;
namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Policy {
    const char* key;std::string old;bool present;
    Policy(const char* k,const char* value):key(k){const char* p=std::getenv(k);present=p!=nullptr;if(p)old=p;set(value);}
    void set(const char* value){
#ifdef _WIN32
        require(_putenv_s(key,value)==0,"cannot set test policy");
#else
        require(setenv(key,value,1)==0,"cannot set test policy");
#endif
    }
    ~Policy(){
#ifdef _WIN32
        _putenv_s(key,present?old.c_str():"");
#else
        if(present)setenv(key,old.c_str(),1);else unsetenv(key);
#endif
    }
};
ModelConfig config(){
    ModelConfig c;c.num_layers=1;c.d_model=64;c.vocab_size=67;c.n_heads=4;c.n_kv_heads=2;
    c.attention_period=64;c.use_moe=true;c.moe_period=1;c.moe_slot=0;
    c.num_experts=4;c.num_experts_per_token=1;c.moe_expert_hidden_dim=128;
    c.use_ttt=false;c.use_chrass=false;c.dropout=0;c.mamba_d_state=8;c.mamba_head_dim=32;c.tie_word_embeddings=false;
    return c;
}
void configure(Trainer& t){t.phase_scheduler.progressive_qat_enabled=false;t.phase_scheduler.ternary_regularization=0;
    t.moe_aux_loss_scale=0;t.weight_decay=0.03f;t.max_grad_norm=1;t.warmup_steps=0;t.total_training_steps=100;t.gradient_accumulation_steps=2;}
void select(JambaModel& model,int expert){
    auto& gate=*model.layers.front()->router->gate;
    Tensor weights=Tensor::zeros(gate.weight.data.shape.dims,Device::CPU);
    for(int i=0;i<model.model_config().d_model;++i)weights.data()[expert*model.model_config().d_model+i]=1;
    // Fixture routing edit preserves master storage/version. It is not a
    // training contribution or a second optimizer updating the registry.
    gate.weight.data.copy_from(weights.to(Device::GPU));
}
std::vector<Parameter*> expert(JambaModel& model,int e){
    auto up=model.layers.front()->expert_gate_up[e]->parameters();auto down=model.layers.front()->expert_down[e]->parameters();
    up.insert(up.end(),down.begin(),down.end());return up;
}
void same(const Tensor& a,const Tensor& b,const char* message){nsos::gpu_parity_test::assert_close(a,b,0,message);}
// Portable checkpoints do not serialize Parameter's cache generation.
// Audit publication against each instance's own pre-commit epoch/activity.
float commit_with_version_audit(Trainer& trainer) {
    const auto parameters = trainer.model->parameters();
    std::vector<uint64_t> before;
    std::vector<unsigned char> contributed;
    {
        gpu::ExecutionContext::Scope lane(trainer.device_sparse_execution_context());
        DeviceGradientAuditScope audit;
        for (auto* parameter : parameters) {
            before.push_back(parameter->version);
            contributed.push_back(parameter->trainable && parameter->data.size && parameter->has_gradient());
        }
    }
    const float loss = trainer.commit_optimizer_step(2);
    require(!trainer.last_optimizer_step_skipped, "valid continuation group was skipped");
    for (size_t i = 0; i < parameters.size(); ++i)
        if (parameters[i]->version != before[i] + contributed[i])
            throw std::runtime_error("continuation version publication differs: " + parameters[i]->name);
    return loss;
}
void trainer_contract(bool cce){
    Policy head("NSOS_HEAD_CCE",cce?"1":"0");
    const auto c=config();JambaModel model(c,Device::GPU);
    model.embedding->weight.copy_data_from(Tensor::ones({c.vocab_size,c.d_model},Device::GPU));
    for(auto* p:model.layers.front()->mamba_layer->parameters())p->copy_data_from(Tensor::zeros(p->data.shape.dims,Device::GPU));
    model.layers.front()->router->gate->set_exact_linear_mode(true);
    Trainer t(&model,2e-4f);configure(t);
    std::vector<int> input(33,1),target(33,2);
    const auto untouched=expert(model,3);std::vector<Tensor> weights;std::vector<uint64_t> versions;
    for(auto* p:untouched){weights.push_back(p->data.clone());versions.push_back(p->version);}
    select(model,0);t.accumulate_microbatch(input,target);
    select(model,1);t.accumulate_microbatch(input,target);
    require(std::isfinite(t.commit_optimizer_step(2)),"Trainer device group rejected");
    require(!t.device_sparse_group_open&&t.global_step_count==1&&t.tokens_committed==66,"Trainer group counters/lifetime differ");
    t.synchronize_device_sparse_checkpoint();
    for(int e:{0,1})for(auto* p:expert(model,e))if(p->trainable)require(t.m_state.count(p)&&t.v_state.count(p),"union lost lazy state");
    for(size_t i=0;i<untouched.size();++i){auto* p=untouched[i];same(p->data,weights[i],"inactive expert changed");require(p->version==versions[i]&&!t.m_state.count(p)&&!t.v_state.count(p),"inactive expert state/version changed");}
    const auto norm=t.last_grad_norm_pre_clip;
    require(std::isfinite(norm)&&t.last_grad_norm_post_clip<=t.max_grad_norm,"Trainer clipping telemetry differs");
    const auto token_count=t.tokens_committed;
    select(model,0);t.accumulate_microbatch(input,target);t.abort_gradient_accumulation();
    require(t.global_step_count==1&&t.tokens_committed==token_count&&!t.device_sparse_group_open&&t.pending_accumulation_microbatches==0,"abort advanced trajectory");
    const auto snapshot=t.capture_checkpoint_snapshot();require(snapshot&&snapshot->global_step()==1,"portable sparse snapshot differs");
    const auto stem=std::filesystem::temp_directory_path()/std::filesystem::path("nsos-device-sparse-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto mp=stem.string()+".model",sp=stem.string()+".state";
    struct Files {std::string a,b;~Files(){std::error_code ec;std::filesystem::remove(a,ec);std::filesystem::remove(b,ec);}} files{mp,sp};
    snapshot->write(mp,sp);
    JambaModel resumed(c,Device::GPU);
    // The fixture recipe is runtime policy, not model weight payload. Restore
    // exact router math/trainability before reading the trainable-only sidecar.
    resumed.layers.front()->router->gate->set_exact_linear_mode(true);
    resumed.load(mp,true);
    const auto source_registry=model.parameters(),resume_registry=resumed.parameters();
    require(source_registry.size()==resume_registry.size(),"resume fixture registry size differs");
    for(size_t i=0;i<source_registry.size();++i)
        if(source_registry[i]->name!=resume_registry[i]->name ||
           source_registry[i]->trainable!=resume_registry[i]->trainable)
            throw std::runtime_error("resume fixture parameter recipe differs: "+source_registry[i]->name);
    Trainer rt(&resumed,2e-4f);configure(rt);rt.load_training_state(sp,mp);
    for(auto* m:{&model,&resumed})select(*m,0);
    for(auto* tr:{&t,&rt}){tr->accumulate_microbatch(input,target);tr->accumulate_microbatch(input,target);require(std::isfinite(commit_with_version_audit(*tr)),"resumed Trainer group rejected");tr->synchronize_device_sparse_checkpoint();}
    const auto a=model.parameters(),b=resumed.parameters();require(a.size()==b.size(),"resumed registry size differs");
    for(size_t i=0;i<a.size();++i){same(a[i]->data,b[i]->data,"checkpoint continuation weights differ");require(a[i]->name==b[i]->name,"checkpoint continuation parameter identity differs");
        if(t.m_state.count(a[i])!=rt.m_state.count(b[i]) || t.v_state.count(a[i])!=rt.v_state.count(b[i]))
            throw std::runtime_error("checkpoint continuation lazy presence differs: "+a[i]->name+
                " original_m="+std::to_string(t.m_state.count(a[i]))+" resumed_m="+std::to_string(rt.m_state.count(b[i]))+
                " original_v="+std::to_string(t.v_state.count(a[i]))+" resumed_v="+std::to_string(rt.v_state.count(b[i]))+
                " original_trainable="+std::to_string(a[i]->trainable)+" resumed_trainable="+std::to_string(b[i]->trainable));
        if(t.m_state.count(a[i])){same(t.m_state.at(a[i]),rt.m_state.at(b[i]),"checkpoint continuation m differs");same(t.v_state.at(a[i]),rt.v_state.at(b[i]),"checkpoint continuation v differs");}}
    require(t.tokens_committed==rt.tokens_committed&&t.global_step_count==rt.global_step_count,"checkpoint continuation counters differ");
}
}
int main(){return nsos::gpu_parity_test::run_parity("sparse_adam_trainer",[]{
    Policy device("NSOS_MOE_DEVICE_ADAM","1"),grouped("NSOS_MOE_GROUPED_TRAINING","1"),ordered("NSOS_MOE_ORDERED_DEVICE","1"),
        crit("NSOS_CRIT_REG","0"),crit_lr("NSOS_CRIT_LR","0");
    const bool old=determinism::deterministic_reductions_enabled();
    struct Restore{bool old;~Restore(){determinism::set_deterministic_reductions(old);}} restore{old};
    determinism::set_deterministic_reductions(true);trainer_contract(false);trainer_contract(true);
});}
