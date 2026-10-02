#include "trainer.h"
#include "nsos/determinism.h"
#include <filesystem>
#include <chrono>
#include <iostream>
#include <limits>
using namespace nsos;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
ModelConfig config(){ModelConfig c;c.num_layers=1;c.d_model=16;c.vocab_size=40;c.n_heads=4;c.n_kv_heads=2;c.attention_period=64;c.use_moe=false;c.use_ttt=false;c.use_chrass=false;c.dropout=0;c.mamba2_faithful=false;c.tie_word_embeddings=false;return c;}
void setup(Trainer& t,bool tokens=true){t.scheduler_unit=tokens?Trainer::SchedulerUnit::Tokens:Trainer::SchedulerUnit::Steps;t.gradient_accumulation_steps=1;t.warmup_tokens=32;t.training_tokens=128;t.warmup_steps=0;t.total_training_steps=10;t.min_learning_rate_scale=1;t.phase_scheduler.progressive_qat_enabled=false;t.dynamic_loss_scaling_enabled=false;t.moe_aux_loss_scale=0;t.weight_decay=0;t.max_grad_norm=1000;t.first_token_loss_scale=1;t.eos_loss_scale=1;}
struct Initial {
    std::string path=(std::filesystem::temp_directory_path()/("nsos-entrypoints-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();
    Initial(JambaModel& m){m.save(path);}~Initial(){std::error_code ec;std::filesystem::remove(path,ec);}
};
void models_same(JambaModel& a,JambaModel& b){const auto pa=a.parameters(),pb=b.parameters();require(pa.size()==pb.size(),"registry mismatch");for(size_t n=0;n<pa.size();++n){auto x=pa[n]->data.cpu(),y=pb[n]->data.cpu();for(int i=0;i<x.size;++i)require(std::isfinite(x.data()[i])&&std::abs(x.data()[i]-y.data()[i])<3e-7,"Tokens LR differs from independent explicit LR");}}
void supervised(){
    JambaModel m(config(),Device::CPU);Initial initial(m);JambaModel oracle(config(),Device::CPU);oracle.load(initial.path,true);
    Trainer t(&m,.02f),ref(&oracle,.02f*8/32);setup(t);setup(ref,false);
    const std::vector<std::vector<int>> prompt{{1,2},{3,4}},answer{{5,6,7},{8,9,10}};
    t.train_supervised_batch(prompt,answer);ref.train_supervised_batch(prompt,answer);
    require(t.tokens_processed==8&&t.tokens_committed==8&&t.pending_accumulated_tokens==0,"supervised token counters omitted");models_same(m,oracle);
}
void loop(){
    JambaModel m(config(),Device::CPU);Initial initial(m);JambaModel oracle(config(),Device::CPU);oracle.load(initial.path,true);
    Trainer t(&m,.02f),ref(&oracle,.02f*4/32);setup(t);setup(ref,false);std::vector<int> tokens{1,2,3,4,5,6,7};
    t.train_loop(tokens,1,2,2,nullptr,2);ref.train_loop(tokens,1,2,2,[&](int step,float){if(step==1)ref.learning_rate=.02f*6/32;},2);
    require(t.tokens_processed==6&&t.tokens_committed==6&&t.global_step_count==2&&t.pending_accumulated_tokens==0,"train_loop token counters omitted");models_same(m,oracle);
}
void pending_guard(){
    JambaModel m(config(),Device::CPU);Trainer t(&m,.02f);setup(t);t.accumulate_microbatch({1,2,3},{2,3,4});
    auto grad=m.embedding->weight.grad.clone();auto processed=t.tokens_processed,pending=t.pending_accumulated_tokens;bool rejected=false;
    try{t.train_supervised_batch({{1,2}},{{3,4}});}catch(const std::logic_error&){rejected=true;}
    require(rejected,"supervised discarded an open accumulation group");require(t.tokens_processed==processed&&t.pending_accumulated_tokens==pending&&t.pending_accumulation_microbatches==1,"guard mutated progress");
    for(int i=0;i<grad.size;++i)require(grad.data()[i]==m.embedding->weight.grad.data()[i],"guard mutated old gradient");t.abort_gradient_accumulation();
}
void overflow(){
    JambaModel m(config(),Device::CPU);Trainer t(&m,.02f);setup(t,false);t.tokens_processed=std::numeric_limits<long long>::max()-1;bool rejected=false;
    try{t.train_supervised_batch({{1,2}},{{3,4}});}catch(const std::overflow_error&){rejected=true;}
    require(rejected&&t.global_step_count==0&&!t.optimizer_state_poisoned(),"counter overflow entered training");
}
void rejection_processed_only(){
#ifdef NSOS_ENABLE_TEST_HOOKS
    JambaModel m(config(),Device::CPU);Trainer t(&m,.02f);setup(t);auto before=m.embedding->weight.data.clone();testing::inject_training_nan_before_optimizer();bool rejected=false;
    try{t.train_supervised_batch({{1,2}},{{3,4}});}catch(const std::runtime_error&){rejected=true;}
    require(rejected&&t.tokens_processed==3&&t.tokens_committed==0&&t.global_step_count==0&&t.pending_accumulated_tokens==0&&!t.optimizer_state_poisoned(),"finite rejection counters/cleanup differ");
    for(int i=0;i<before.size;++i)require(before.data()[i]==m.embedding->weight.data.data()[i],"rejected group changed weights");
    t.train_supervised_batch({{1,2}},{{3,4}});require(t.tokens_processed==6&&t.tokens_committed==3&&t.global_step_count==1,"recovery counters differ");
#endif
}
}
int main(){determinism::set_deterministic_reductions(true);int failures=0;for(auto test:{std::pair{"supervised_counters_and_tokens_lr",supervised},std::pair{"loop_counters_tail_and_tokens_lr",loop},std::pair{"pending_group_guard",pending_guard},std::pair{"counter_overflow_before_forward",overflow},std::pair{"finite_reject_processed_only_and_retry",rejection_processed_only}}){try{test.second();std::cout<<test.first<<" PASS\n";}catch(const std::exception& e){++failures;std::cout<<test.first<<" FAIL "<<e.what()<<'\n';}}return failures?1:0;}
