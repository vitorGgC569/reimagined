#include "trainer.h"
#include "nsos/sha256.h"
#include "nsos/determinism.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace nsos;
namespace {
void require(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
struct Files {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("nsos-progress-v11-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Files(){std::filesystem::create_directories(directory);}
    ~Files(){std::error_code e;std::filesystem::remove_all(directory,e);}
    std::string path(const char* name)const{return (directory/name).string();}
};
std::vector<char> read(const std::string& p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
void write(const std::string& p,const std::vector<char>& b){std::ofstream f(p,std::ios::binary);f.write(b.data(),b.size());require(bool(f),"fixture write failed");}
template<class T> void put(std::vector<char>& b,size_t offset,T value){require(offset+sizeof(T)<=b.size(),"fixture offset invalid");std::memcpy(b.data()+offset,&value,sizeof(T));}
template<class T> T get(const std::vector<char>& b,size_t offset){T result;require(offset+sizeof(T)<=b.size(),"fixture read offset invalid");std::memcpy(&result,b.data()+offset,sizeof(T));return result;}
// v11 progress uses explicit POD fields: TPR1 magic + 232 bytes, no padding.
constexpr size_t trailer_bytes=4+8+64,progress_bytes=236;
size_t progress_offset(const std::vector<char>& b){require(b.size()>trailer_bytes+progress_bytes,"missing progress fixture");const auto offset=b.size()-trailer_bytes-progress_bytes;require(get<uint32_t>(b,offset)==0x31525054u,"missing TPR1 marker");return offset;}
void resign(std::vector<char>& b){const auto payload=b.size()-trailer_bytes;put<uint64_t>(b,payload+4,payload);const auto sha=integrity::sha256_hex(b.data(),payload);std::copy(sha.begin(),sha.end(),b.end()-64);}
std::vector<char> legacy10(std::vector<char> b){const auto start=progress_offset(b);b.erase(b.begin()+start,b.begin()+start+progress_bytes);put<uint32_t>(b,4,10);resign(b);return b;}
ModelConfig config(){ModelConfig c;c.num_layers=1;c.d_model=16;c.vocab_size=40;c.n_heads=4;c.n_kv_heads=2;c.attention_period=64;c.use_moe=false;c.use_ttt=false;c.use_chrass=false;c.dropout=0;c.mamba2_faithful=false;c.tie_word_embeddings=false;return c;}
void setup(Trainer& t,bool tokens){t.scheduler_unit=tokens?Trainer::SchedulerUnit::Tokens:Trainer::SchedulerUnit::Steps;t.gradient_accumulation_steps=2;t.warmup_tokens=12;t.training_tokens=48;t.decay_tokens=16;t.warmup_steps=2;t.total_training_steps=9;t.phase_scheduler.progressive_qat_enabled=false;t.phase_scheduler.ternary_regularization=0;t.dynamic_loss_scaling_enabled=false;t.weight_decay=.03f;t.max_grad_norm=.1f;t.first_token_loss_scale=1;t.eos_loss_scale=1;t.moe_aux_loss_scale=0;}
const std::vector<int> x{1,3,5,7},y{3,5,7,9},xx{2,4,6},yy{4,6,8};
float group(Trainer& t){(void)t.accumulate_microbatch(x,y);(void)t.accumulate_microbatch(xx,yy);auto loss=t.commit_optimizer_step(2);require(std::isfinite(loss)&&!t.last_optimizer_step_skipped,"valid CPU A2 group rejected");return loss;}
void tensor_same(const Tensor& a,const Tensor& b,const std::string& label){require(a.shape==b.shape&&a.size==b.size&&std::memcmp(a.data(),b.data(),a.size*sizeof(float))==0,label);}
void same(Trainer& a,Trainer& b,bool counters=true){const auto p=a.model->parameters(),q=b.model->parameters();require(p.size()==q.size(),"registry size differs");for(size_t i=0;i<p.size();++i){require(p[i]->name==q[i]->name&&p[i]->trainable==q[i]->trainable,"registry recipe differs");tensor_same(p[i]->data,q[i]->data,"weights differ: "+p[i]->name);require(a.m_state.count(p[i])==b.m_state.count(q[i])&&a.v_state.count(p[i])==b.v_state.count(q[i]),"lazy differs: "+p[i]->name);if(a.m_state.count(p[i])){tensor_same(a.m_state.at(p[i]),b.m_state.at(q[i]),"m differs");tensor_same(a.v_state.at(p[i]),b.v_state.at(q[i]),"v differs");}}
    require(a.global_step_count==b.global_step_count&&a.gradient_accumulation_steps==b.gradient_accumulation_steps&&a.scheduler_unit==b.scheduler_unit,"step/A/scheduler differs");
    if(counters)require(a.tokens_processed==b.tokens_processed&&a.tokens_committed==b.tokens_committed&&a.token_counters_complete==b.token_counters_complete,"token counters/history differ");
    require(a.last_grad_norm_pre_clip==b.last_grad_norm_pre_clip&&a.last_grad_norm_post_clip==b.last_grad_norm_post_clip&&a.last_update_was_clipped==b.last_update_was_clipped&&a.last_accumulation_steps==b.last_accumulation_steps&&a.last_optimizer_step_skipped==b.last_optimizer_step_skipped,"commit telemetry differs");
}
std::vector<char> export_state(Trainer& t,const Files& files,const std::string& model){const auto output=files.path("audit.state");t.save_training_state(output,model);return read(output);}
void telemetry_fixture(Trainer& t){t.last_auxiliary_stats.prompt_tokens=3;t.last_auxiliary_stats.reason_cosine=-.25f;t.last_objective_stats.sparse_selector=.125f;t.last_step_telemetry.enabled=true;t.last_step_telemetry.global_step=t.global_step_count;t.last_step_telemetry.bucket_count=2;t.last_step_telemetry.wall_ms=1.5;t.last_step_telemetry.unaccounted_ms=-.125;}
template<class F> std::string rejected(F f,const std::string& why){try{f();}catch(const std::exception& e){return e.what();}throw std::runtime_error("accepted "+why);}
void continuation(bool tokens,bool snapshot){Files files;JambaModel m(config(),Device::CPU);Trainer t(&m,.002f);setup(t,tokens);group(t);group(t);
    (void)t.accumulate_microbatch(x,y);t.abort_gradient_accumulation(); // processed != committed is authoritative
    require(t.tokens_processed==18&&t.tokens_committed==14,"fixture counters invalid");telemetry_fixture(t);
    auto model=files.path("model"),state=files.path("state");
    if(snapshot){auto captured=t.capture_checkpoint_snapshot();t.tokens_processed+=11;captured->write(model,state);t.tokens_processed-=11;}
    else{m.save(model);t.save_training_state(state,model);}
    require(get<uint32_t>(read(state),4)==11,"writer did not emit v11");
    JambaModel r(config(),Device::CPU);r.load(model,true);Trainer rt(&r,.02f);setup(rt,tokens);
    rt.warmup_tokens=1;rt.training_tokens=200;rt.decay_tokens=0; // scalar schedule restored, structural unit/A must match
    rt.load_training_state(state,model);same(t,rt);
    require(read(state)==export_state(rt,files,model),"roundtrip omitted metadata/telemetry/RNG");
    for(int i=0;i<5;++i){require(group(t)==group(rt),"continuation loss differs");same(t,rt);}
    std::cout<<"PASS exact CPU A2 continuation "<<(tokens?"Tokens":"Steps")<<" "<<(snapshot?"snapshot":"direct")<<'\n';
}
void rejection_and_legacy(bool tokens){Files files;JambaModel m(config(),Device::CPU);Trainer t(&m,.002f);setup(t,tokens);group(t);telemetry_fixture(t);const auto model=files.path("model"),state=files.path("state");m.save(model);t.save_training_state(state,model);const auto valid=read(state);
    const auto original=export_state(t,files,model);const auto identity=t.execution_identity_digest();
    auto unchanged=[&]{require(export_state(t,files,model)==original,"rejected load mutated Trainer metadata/moments/RNG");require(t.execution_identity_digest()==identity&&!t.optimizer_state_poisoned(),"rejected load mutated identity/poison");};
    auto corrupt=[&](size_t offset,auto value,const std::string& label){auto bytes=valid;put(bytes,progress_offset(bytes)+offset,value);resign(bytes);write(state,bytes);auto message=rejected([&]{t.load_training_state(state,model);},label);require(message.find("progress")!=std::string::npos||message.find("telemetry")!=std::string::npos||message.find("counter")!=std::string::npos,"wrong semantic rejection: "+message);unchanged();};
    corrupt(4,uint32_t{2},"unknown scheduler");corrupt(8,int32_t{0},"invalid declared A");corrupt(8,int32_t{3},"mismatched A with unchanged identity");corrupt(36,int64_t{-1},"negative tokens");corrupt(44,int64_t{8},"committed above processed");corrupt(52,uint8_t{2},"invalid completeness bool");corrupt(53,std::numeric_limits<float>::quiet_NaN(),"NaN clipping telemetry");corrupt(62,int32_t{0},"invalid last A");
    if(tokens){corrupt(52,uint8_t{0},"incomplete token scheduler");corrupt(20,int64_t{0},"invalid token schedule");corrupt(28,int64_t{49},"invalid decay window");}
    auto truncated=valid;auto pos=progress_offset(truncated);truncated.erase(truncated.begin()+pos,truncated.begin()+pos+progress_bytes);resign(truncated);write(state,truncated);rejected([&]{t.load_training_state(state,model);},"v11 missing progress block");unchanged();
    auto badsha=valid;badsha[progress_offset(badsha)+36]^=1;write(state,badsha);rejected([&]{t.load_training_state(state,model);},"bad SHA");unchanged();
    write(state,valid);
#ifdef NSOS_ENABLE_TEST_HOOKS
    testing::set_training_state_stage_failure_countdown(0);rejected([&]{t.load_training_state(state,model);},"allocation failure");testing::clear_training_state_stage_failure();unchanged();
    testing::set_training_state_save_failure_before_replace(true);rejected([&]{t.save_training_state(state,model);},"interrupted save");testing::set_training_state_save_failure_before_replace(false);require(read(state)==valid,"interrupted save replaced sidecar");
#endif
    JambaModel wrong(config(),Device::CPU);wrong.load(model,true);Trainer wt(&wrong,.002f);setup(wt,tokens);wt.gradient_accumulation_steps=1;rejected([&]{wt.load_training_state(state,model);},"runtime identity A mismatch");require(wt.tokens_committed==0&&wt.m_state.empty(),"A mismatch staged state");wt.gradient_accumulation_steps=2;wt.scheduler_unit=tokens?Trainer::SchedulerUnit::Steps:Trainer::SchedulerUnit::Tokens;rejected([&]{wt.load_training_state(state,model);},"runtime identity scheduler mismatch");
    const auto v10=legacy10(valid);write(state,v10);rejected([&]{t.load_training_state(state,model);},"implicit legacy migration");unchanged();
    if(tokens){rejected([&]{t.load_training_state(state,model,true,true);},"legacy token scheduler even with both permissions");unchanged();}
    else{t.load_training_state(state,model,false,true);require(!t.token_counters_complete&&t.tokens_processed==0&&t.tokens_committed==0&&t.global_step_count==1&&t.gradient_accumulation_steps==2,"legacy migration invented historical counters");require(t.m_state.size()>0,"legacy migration lost moments");group(t);require(t.tokens_processed==7&&t.tokens_committed==7&&!t.token_counters_complete,"legacy migrated counters not explicit lower bound");
        const auto migrated=files.path("migrated"),migrated_state=files.path("migrated.state");auto snap=t.capture_checkpoint_snapshot();snap->write(migrated,migrated_state);JambaModel restored(config(),Device::CPU);restored.load(migrated,true);Trainer r(&restored,.002f);setup(r,false);r.load_training_state(migrated_state,migrated);same(t,r);require(!r.token_counters_complete,"snapshot lost incomplete-history flag");r.scheduler_unit=Trainer::SchedulerUnit::Tokens;rejected([&]{group(r);},"incomplete legacy switched to token scheduler");}
    std::cout<<"PASS rollback, corrupt and explicit legacy "<<(tokens?"Tokens":"Steps")<<'\n';
}
void boundaries(){Files files;JambaModel m(config(),Device::CPU);Trainer t(&m,.002f);setup(t,false);group(t);const auto model=files.path("model"),state=files.path("state");m.save(model);t.save_training_state(state,model);const auto durable=read(state);(void)t.accumulate_microbatch(x,y);const auto processed=t.tokens_processed,committed=t.tokens_committed,pending=t.pending_accumulated_tokens;auto grad=m.embedding->weight.grad.clone();
    rejected([&]{t.save_training_state(state,model);},"save pending CPU group");require(read(state)==durable,"pending save replaced sidecar");rejected([&]{(void)t.capture_checkpoint_snapshot();},"snapshot pending CPU group");rejected([&]{t.load_training_state(state,model);},"load pending CPU group");require(t.tokens_processed==processed&&t.tokens_committed==committed&&t.pending_accumulated_tokens==pending&&t.pending_accumulation_microbatches==1,"boundary rejection mutated progress");tensor_same(grad,m.embedding->weight.grad,"boundary rejection mutated gradient");t.abort_gradient_accumulation();t.save_training_state(state,model);t.load_training_state(state,model);require(t.tokens_processed==processed&&t.tokens_committed==committed,"abort checkpoint lost processed-only tokens");
    t.tokens_processed=std::numeric_limits<long long>::max()-1;rejected([&]{(void)t.accumulate_microbatch(x,y);},"token counter overflow before forward");require(t.pending_accumulation_microbatches==0&&!t.optimizer_state_poisoned(),"overflow entered accumulation/poisoned optimizer");
    std::cout<<"PASS closed boundaries and token overflow\n";
}
void single_step_contract() {
    Files files;
    JambaModel direct_model(config(), Device::CPU);
    const auto model_path = files.path("initial.model");
    direct_model.save(model_path);
    JambaModel accumulated_model(config(), Device::CPU);
    accumulated_model.load(model_path, true);
    Trainer direct(&direct_model, .002f), accumulated(&accumulated_model, .002f);
    setup(direct, true);
    setup(accumulated, true);
    direct.gradient_accumulation_steps = accumulated.gradient_accumulation_steps = 1;
    for (int index = 0; index < 5; ++index) {
        const float direct_loss = direct.train_step(x, y);
        (void)accumulated.accumulate_microbatch(x, y);
        const float accumulated_loss = accumulated.commit_optimizer_step(1);
        require(direct_loss == accumulated_loss, "Tokens A1 entry points have different losses");
        same(direct, accumulated);
        require(direct.pending_accumulated_tokens == 0, "train_step retained pending token contribution");
    }
    auto before = direct_model.embedding->weight.data.clone();
    direct.gradient_accumulation_steps = 2;
    const auto processed = direct.tokens_processed, committed = direct.tokens_committed;
    rejected([&] { (void)direct.train_step(x, y); }, "train_step under declared A2 identity");
    tensor_same(before, direct_model.embedding->weight.data, "A2 rejection changed weights");
    require(direct.tokens_processed == processed && direct.tokens_committed == committed,
            "A2 rejection changed counters");
    direct.gradient_accumulation_steps = 1;
    (void)direct.accumulate_microbatch(x, y);
    const auto pending = direct.pending_accumulated_tokens;
    rejected([&] { (void)direct.train_step(xx, yy); }, "train_step over an existing accumulation group");
    require(direct.pending_accumulated_tokens == pending && direct.pending_accumulation_microbatches == 1,
            "pending-group rejection changed the existing group");
    direct.abort_gradient_accumulation();
    std::cout << "PASS Tokens single-step/A1 parity and declared accumulation guard\n";
}

void poisoned_open_owner_contract() {
    Files files;
    JambaModel model(config(), Device::CPU);
    Trainer trainer(&model, .002f);
    setup(trainer, false);
    group(trainer);
    const auto model_path = files.path("model"), state_path = files.path("state");
    model.save(model_path);
    trainer.save_training_state(state_path, model_path);
    auto before = model.embedding->weight.data.clone();
    const auto committed = trainer.tokens_committed;
    // Host contract simulation; this does not claim GPU rollback fault coverage.
    trainer.device_sparse_group_open = true;
    trainer.mark_optimizer_state_poisoned();
    const auto message = rejected([&] { trainer.load_training_state(state_path, model_path); },
                                  "in-place reload with an unclosed poisoned owner");
    require(message.find("fresh model and Trainer") != std::string::npos,
            "poisoned owner did not identify the required fresh-owner recovery");
    tensor_same(before, model.embedding->weight.data, "poison rejection changed weights");
    require(trainer.optimizer_state_poisoned() && trainer.device_sparse_group_open &&
            trainer.tokens_committed == committed, "poison rejection changed owner/progress");
    JambaModel fresh_model(config(), Device::CPU);
    fresh_model.load(model_path, true);
    Trainer fresh(&fresh_model, .002f);
    setup(fresh, false);
    fresh.load_training_state(state_path, model_path);
    require(!fresh.optimizer_state_poisoned() && fresh.tokens_committed == committed,
            "fresh owner did not restore the authenticated trajectory");
    trainer.device_sparse_group_open = false;
    std::cout << "PASS explicit fresh-owner recovery contract (host simulation)\n";
}
}
int main(){try{for(bool tokens:{false,true})for(bool snapshot:{false,true})continuation(tokens,snapshot);rejection_and_legacy(false);rejection_and_legacy(true);boundaries();single_step_contract();poisoned_open_owner_contract();return 0;}catch(const std::exception& e){
#ifdef NSOS_ENABLE_TEST_HOOKS
    testing::clear_training_state_stage_failure();testing::set_training_state_save_failure_before_replace(false);
#endif
    std::cerr<<"FAIL progress checkpoint v11: "<<e.what()<<'\n';return 1;}}
