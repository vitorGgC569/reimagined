#include "gpu_parity_common.h"
#include "trainer.h"
#include "nsos/determinism.h"
#include <filesystem>
#include <chrono>
#include <array>
#include <iostream>
#include <limits>
#include <algorithm>
using namespace nsos;
namespace {
void require(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
struct Policy {const char* key;std::string old;bool present;Policy(const char* k,const char* v):key(k){const char* x=std::getenv(k);present=x;if(x)old=x;
#ifdef _WIN32
_putenv_s(k,v);
#else
setenv(k,v,1);
#endif
}~Policy(){
#ifdef _WIN32
_putenv_s(key,present?old.c_str():"");
#else
if(present)setenv(key,old.c_str(),1);else unsetenv(key);
#endif
}};
ModelConfig config(){ModelConfig c;c.num_layers=1;c.d_model=64;c.vocab_size=67;c.n_heads=4;c.n_kv_heads=2;c.attention_period=64;c.use_moe=true;c.moe_period=1;c.moe_slot=0;c.num_experts=4;c.num_experts_per_token=1;c.moe_expert_hidden_dim=128;c.use_ttt=false;c.use_chrass=false;c.dropout=0;c.mamba_d_state=8;c.mamba_head_dim=32;c.tie_word_embeddings=true;return c;}
void setup(Trainer& t,float scale,bool qat){t.gradient_accumulation_steps=2;t.loss_scale=scale;t.loss_scale_growth_interval=10000;t.dynamic_loss_scaling_enabled=true;t.moe_aux_loss_scale=0;t.max_grad_norm=1000;t.weight_decay=.03f;t.warmup_steps=0;t.total_training_steps=100;t.phase_scheduler.progressive_qat_enabled=qat;t.phase_scheduler.semantic_warmup_steps=0;t.phase_scheduler.qat_start_step=2;t.phase_scheduler.ternary_regularization=qat?.125f:0;t.global_step_count=qat?1:0;t.scheduler_unit=Trainer::SchedulerUnit::Tokens;t.warmup_tokens=20;t.training_tokens=200;}
void configure_routing(JambaModel& model){
    auto& gate=*model.layers.front()->router->gate;
    Tensor w=Tensor::zeros(gate.weight.data.shape.dims,Device::CPU);
    for(int i=0;i<64;++i){w.data()[i]=1;w.data()[64+i]=-1;}
    gate.weight.copy_data_from(w.to(Device::GPU));
}
struct Files {std::string stem=(std::filesystem::temp_directory_path()/("nsos-loss-scale-a2-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();~Files(){for(auto suffix:{".model",".state",".initial"}){std::error_code ec;std::filesystem::remove(stem+suffix,ec);}}};
struct Contribution {bool active=false;Tensor grad;};
std::vector<Contribution> gradients(Trainer& t){gpu::ExecutionContext::Scope lane(t.device_sparse_execution_context());DeviceGradientAuditScope audit;std::vector<Contribution> out;for(auto* p:t.model->parameters()){bool active=p->trainable&&p->has_gradient();out.push_back({active,active?p->grad.cpu().clone():Tensor()});}return out;}
void same_weights(JambaModel& a,JambaModel& b,float atol=0){auto ap=a.parameters(),bp=b.parameters();require(ap.size()==bp.size(),"registry count");for(size_t i=0;i<ap.size();++i)gpu_parity_test::assert_close(ap[i]->data,bp[i]->data,atol,("weights "+ap[i]->name).c_str());}

void require_expert_membership(JambaModel& model,const std::vector<Contribution>& bank,
                               bool first,bool second,const char* label){
    const auto registry=model.parameters();auto& block=*model.layers.front();
    for(size_t e=0;e<block.expert_gate_up.size();++e){
        const bool expected=(e==0&&first)||(e==1&&second);
        for(auto* weight:{&block.expert_gate_up[e]->weight,&block.expert_down[e]->weight}){
            require(weight->has_device_gradient_activity(),"expert weight is not device-bound");
            auto it=std::find(registry.begin(),registry.end(),weight);
            require(it!=registry.end(),"expert weight missing canonical registry");
            const size_t i=size_t(it-registry.begin());
            require(i<bank.size()&&bank[i].active==expected,
                    std::string(label)+" expert="+std::to_string(e)+" "+weight->name);
        }
    }
}
struct FrozenParameterBank{
    std::vector<Parameter*> parameters;
    std::vector<uint64_t> versions;
    std::vector<float*> storage;
    std::vector<Tensor> values;
    explicit FrozenParameterBank(JambaModel& model):parameters(model.parameters()){
        for(auto* p:parameters){versions.push_back(p->version);storage.push_back(p->data.raw_data());values.push_back(p->data.cpu().clone());}
    }
    void check(JambaModel& model,const char* where)const{
        require(model.parameters()==parameters,"canonical registry changed within group");
        for(size_t i=0;i<parameters.size();++i){
            auto* p=parameters[i];
            require(p->version==versions[i]&&p->data.raw_data()==storage[i],std::string(where)+" version/storage "+p->name);
            gpu_parity_test::assert_close(p->data,values[i],0,(std::string(where)+" values "+p->name).c_str(),0);
        }
    }
};


void zero_previous_contributions_async(Trainer& trainer,const std::vector<Contribution>& mask){
#ifdef USE_CUDA
    gpu::ExecutionContext::Scope lane(trainer.device_sparse_execution_context());
    const auto params=trainer.model->parameters();
    require(params.size()==mask.size(),"zero mask registry mismatch");
    for(size_t i=0;i<mask.size();++i)if(mask[i].active){
        auto* p=params[i];
        require(p->grad.size>0&&p->grad.get_device()==Device::GPU,"zero previous missing device gradient");
        const auto status=cudaMemsetAsync(p->grad.raw_data(),0,size_t(p->grad.size)*sizeof(float),gpu::current_stream());
        require(status==cudaSuccess,"zero previous async enqueue failed");
    }
    // Numerical zero is still a contribution. Never clear host/device
    // activity, unbind an owner, or synchronize/materialize gradients here.
#else
    throw std::runtime_error("unobserved zero requires GPU");
#endif
}

void pair_oracle(float scale,bool qat,bool second_zero,bool cce){
    std::cout<<"A2 begin lossScale="<<scale<<" QAT="<<qat<<" previous_zero="<<second_zero<<" CCE="<<cce<<std::endl;
    Policy head("NSOS_HEAD_CCE",cce?"1":"0");Files files;auto c=config();JambaModel accumulated(c,Device::GPU);
    Tensor embeddings=Tensor::ones({c.vocab_size,c.d_model},Device::CPU);
    for(int i=0;i<c.d_model;++i)embeddings.data()[4*c.d_model+i]=-1;
    accumulated.embedding->weight.copy_data_from(embeddings.to(Device::GPU));
    for(auto* p:accumulated.layers.front()->mamba_layer->parameters())p->copy_data_from(Tensor::zeros(p->data.shape.dims,Device::GPU));
    accumulated.layers.front()->router->gate->set_exact_linear_mode(true);
    configure_routing(accumulated);accumulated.save(files.stem+".initial");
    JambaModel first(c,Device::GPU),second(c,Device::GPU),reference(c,Device::GPU);
    for(auto* m:{&first,&second,&reference}){m->layers.front()->router->gate->set_exact_linear_mode(true);m->load(files.stem+".initial",true);}
    // Prepare canonical Mamba2 packed views before opening/freezing the bank.
    for(auto* m:{&accumulated,&first,&second,&reference}){
        for(auto* linear:m->collect_bitlinear_layers())linear->set_reference_path(true);
        m->set_training_mode(true);
        (void)m->layers.front()->mamba_layer->forward(Tensor::ones({1,5,c.d_model},Device::GPU));
        m->reset_session();
    }
    Trainer t(&accumulated,.002f),a(&first,.002f),b(&second,.002f),ref(&reference,.002f);for(auto* tr:{&t,&a,&b,&ref})setup(*tr,scale,qat);
    // Opposite embedding signs select different experts through fixed weights.
    // Updating router parameters inside an open accumulation group is invalid.
    std::vector<int> x(5,1),xx(5,4),y(5,2),yy(5,3);
    FrozenParameterBank t_before(accumulated),ref_before(reference);
    a.accumulate_microbatch(x,y);auto ga=gradients(a);
    require_expert_membership(first,ga,true,false,"independent positive input");
    b.accumulate_microbatch(xx,yy);auto gb=gradients(b);
    require_expert_membership(second,gb,false,true,"independent negative input");
    t.accumulate_microbatch(x,y);ref.accumulate_microbatch(x,y);
    // No fixture gradient materialization or first-micro oracle read on
    // accumulated/reference models until both VJPs have finished.
    if(second_zero){
        // A zero previous contribution must still be preserved as contributed.
        // This gives a controlled nonzero+zero boundary without changing the
        // model loss implementation or pretending a CE gradient is zero.
        for(auto& contribution:ga)if(contribution.active)std::fill_n(contribution.grad.data(),contribution.grad.size,0.0f);
        zero_previous_contributions_async(t,ga);
        zero_previous_contributions_async(ref,ga);
    }
    t.accumulate_microbatch(xx,yy);ref.accumulate_microbatch(xx,yy);
    auto observed=gradients(t);auto rp=reference.parameters();size_t checked_dense=0,checked_sparse=0;
    require(t.pending_accumulation_microbatches==2&&t.pending_accumulated_tokens==10,"A2 pending counters");
    require_expert_membership(accumulated,observed,true,true,"actual A2 union (zero remains active)");
    t_before.check(accumulated,"after both VJPs");ref_before.check(reference,"reference after both VJPs");
    for(size_t i=0;i<ga.size();++i){
        const bool active=ga[i].active||gb[i].active;require(observed[i].active==active,"activity union mismatch");if(!active)continue;
        Tensor expected=ga[i].active?ga[i].grad.clone():Tensor::zeros(gb[i].grad.shape.dims,Device::CPU);if(gb[i].active)expected.add_inplace_(gb[i].grad);
        gpu_parity_test::assert_close(observed[i].grad,expected,3e-5f,("two-contribution unscale "+rp[i]->name).c_str(),2e-4f);
        rp[i]->grad.copy_from(expected.to(Device::GPU));
        if(rp[i]->has_device_gradient_activity())++checked_sparse;else ++checked_dense;
    }
    require(checked_dense&&checked_sparse,"fixture did not cover dense and sparse contribution");a.abort_gradient_accumulation();b.abort_gradient_accumulation();
    t.commit_optimizer_step(2);ref.commit_optimizer_step(2);require(!t.last_optimizer_step_skipped&&!ref.last_optimizer_step_skipped,"valid group skipped");same_weights(accumulated,reference,3e-6f);
    t.synchronize_device_sparse_checkpoint();ref.synchronize_device_sparse_checkpoint();require(t.tokens_committed==10&&t.tokens_processed==10&&t.pending_accumulated_tokens==0,"A2 counters");
    require(!qat||t.last_objective_stats.qat_regularization>0,"QAT fixture not exercised");
    // Abort keeps processed telemetry and discards the candidate group; a fresh
    // pair must still match the independent contribution oracle.
    const auto committed=t.tokens_committed;t.accumulate_microbatch(x,y);t.abort_gradient_accumulation();require(t.tokens_committed==committed&&t.pending_accumulation_microbatches==0&&!t.device_sparse_group_open,"abort committed candidate");
    auto snapshot=t.capture_checkpoint_snapshot();snapshot->write(files.stem+".model",files.stem+".state");JambaModel resumed(c,Device::GPU);resumed.layers.front()->router->gate->set_exact_linear_mode(true);resumed.load(files.stem+".model",true);Trainer rt(&resumed,.002f);setup(rt,scale,qat);rt.load_training_state(files.stem+".state",files.stem+".model");
    for(auto* tr:{&t,&rt}){tr->accumulate_microbatch(x,y);tr->accumulate_microbatch(xx,yy);tr->commit_optimizer_step(2);}same_weights(accumulated,resumed);require(t.tokens_processed==rt.tokens_processed&&t.tokens_committed==rt.tokens_committed&&t.loss_scale==rt.loss_scale,"checkpoint continuation counters/scaler");
#ifdef NSOS_ENABLE_TEST_HOOKS
    const auto before_tokens=t.tokens_committed;const auto before_step=t.global_step_count;
    t.accumulate_microbatch(x,y);t.accumulate_microbatch(xx,yy);
    // The rollback baseline follows both VJPs and precedes the optimizer.
    std::vector<Tensor> before;for(auto* p:accumulated.parameters())before.push_back(p->data.cpu().clone());
    testing::inject_training_nan_before_optimizer();t.commit_optimizer_step(2);
    require(t.last_optimizer_step_skipped&&t.tokens_committed==before_tokens&&t.global_step_count==before_step&&t.pending_accumulation_microbatches==0&&!t.device_sparse_group_open&&!t.optimizer_state_poisoned(),"global finitegate failed closed");auto p=accumulated.parameters();for(size_t i=0;i<p.size();++i)gpu_parity_test::assert_close(p[i]->data,before[i],0,"finitegate changed weights");
    t.accumulate_microbatch(x,y);t.accumulate_microbatch(xx,yy);t.commit_optimizer_step(2);require(!t.last_optimizer_step_skipped&&t.tokens_committed==before_tokens+10&&t.global_step_count==before_step+1,"fresh group failed poison recovery");
#endif
    std::cout<<"A2 oracle lossScale="<<scale<<" QAT="<<qat<<" previous_zero="<<second_zero<<" CCE="<<cce<<" dense="<<checked_dense<<" sparse="<<checked_sparse<<" PASS\n";
}
}
int main(){return gpu_parity_test::run_parity("loss_scale_accumulation_independent",[]{
#ifdef USE_CUDA
    Policy optimizer("NSOS_OPTIMIZER","adamw"),device("NSOS_MOE_DEVICE_ADAM","1"),grouped("NSOS_MOE_GROUPED_TRAINING","1"),ordered("NSOS_MOE_ORDERED_DEVICE","1"),crit("NSOS_CRIT_REG","0"),cl("NSOS_CRIT_LR","0");
    const auto old_precision=matmul_precision_mode();const bool old_determinism=determinism::deterministic_reductions_enabled();struct Restore {int precision;bool deterministic;~Restore(){set_matmul_precision_mode(precision);determinism::set_deterministic_reductions(deterministic);}} restore{old_precision,old_determinism};set_matmul_precision_mode(2);determinism::set_deterministic_reductions(true);
    for(float scale:{8.0f,8192.0f})for(bool cce:{false,true}){pair_oracle(scale,false,false,cce);pair_oracle(scale,true,false,cce);pair_oracle(scale,false,true,cce);}
#else
    throw std::runtime_error("FP16 accumulation fixture requires GPU backend");
#endif
});}
