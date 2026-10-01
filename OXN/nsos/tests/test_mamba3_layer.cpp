#include "mamba3_layer.h"
#include "mamba3_reference.h"
#include "jamba.h"
#include "trainer.h"
#include "nsos_sdk.h"
#include "nsos_serializer.h"
#include <cmath>
#include <iostream>
#include <filesystem>
#include <thread>
#include <atomic>
#include <limits>

using namespace nsos;
namespace mb=nsos::mamba3_block;
namespace ref=nsos::mamba3_reference;
namespace {
void require(bool b,const char* msg) {if(!b) throw std::runtime_error(msg);}
template<class Fn> void rejected(Fn f,const char* msg) {bool bad=false;try {f();} catch(const std::exception&) {bad=true;}require(bad,msg);}
std::vector<double> values(const Tensor& t) {Tensor h=t.cpu();return {h.data(),h.data()+h.size};}
void close(const Tensor& t,const std::vector<double>& expected,const char* msg,double tol=3e-4) {auto got=values(t);require(got.size()==expected.size(),"close extent");for(std::size_t i=0;i<got.size();++i) if(!std::isfinite(got[i])||std::abs(got[i]-expected[i])>tol*(1+std::abs(expected[i]))) throw std::runtime_error(std::string(msg)+" at "+std::to_string(i)+" got="+std::to_string(got[i])+" expected="+std::to_string(expected[i]));}
void close(const Tensor& a,const Tensor& b,const char* msg,double tol=3e-4) {close(a,values(b),msg,tol);}
void fill(Tensor& t,float scale=.13f,float bias=0) {require(t.get_device()==Device::CPU,"fixture CPU");for(std::size_t i=0;i<std::size_t(t.size);++i) t.data()[i]=bias+scale*std::sin(.173f*(i+1));}
Mamba3Config config(bool mimo,bool norm,int N=128,int rank=4) {Mamba3Config c;c.expand=2;c.head_dim=2;c.state_dim=N;c.n_groups=2;c.mimo=mimo;c.mimo_rank=rank;c.outproj_norm=norm;return c;}
mb::Shape shape(const Mamba3Config& c,int B,int S) {return {B,S,4,4,c.n_groups,c.head_dim,c.state_dim,c.mimo?c.mimo_rank:1,int(c.state_dim*c.rope_fraction)/2,c.mimo,c.outproj_norm,c.norm_eps,c.a_floor};}
Mamba3State state(mb::Shape s,Device dev) {Mamba3State x{Tensor::zeros({s.batch,s.heads,s.rotary_pairs}),Tensor::zeros({s.batch,s.heads,s.head_dim,s.state_dim}),Tensor::zeros({s.batch,s.heads,s.rank,s.state_dim}),Tensor::zeros({s.batch,s.heads,s.head_dim})};for(auto* t:{&x.phase,&x.ssm,&x.k,&x.v}) fill(*t,.003f,.01f);return {x.phase.to(dev),x.ssm.to(dev),x.k.to(dev),x.v.to(dev)};}
ref::State reference(const Mamba3State& s) {return {values(s.phase),values(s.ssm),values(s.k),values(s.v)};}
void close_state(const Mamba3State& a,const ref::State& b) {close(a.phase,b.phase,"phase");close(a.ssm,b.ssm,"SSM");close(a.k,b.k,"previous K");close(a.v,b.v,"raw V");}
std::vector<double> linear(const std::vector<double>& x,const std::vector<double>& w,int rows,int K,int O) {std::vector<double> y(std::size_t(rows)*O);for(int t=0;t<rows;++t) for(int o=0;o<O;++o) for(int k=0;k<K;++k) y[t*O+o]+=x[t*K+k]*w[o*K+k];return y;}
void oracle(bool mimo,bool norm,Device dev) {
    auto c=config(mimo,norm);auto s=shape(c,2,3);Mamba3Layer layer(4,c);layer.to(dev);auto params=layer.parameters();Tensor input({2,3,4});fill(input);for(int i=12;i<24;++i) input.data()[i]=std::numeric_limits<float>::quiet_NaN();
    auto initial=state(s,dev),seed=state(s,dev);std::vector<int> valid{3,0};auto input_values=values(input);for(int i=12;i<24;++i) input_values[i]=0;
    const auto win=values(params[0]->data),wout=values(params[1]->data);ref::BlockInputs x{s,linear(input_values,win,6,4,s.width())};
    for(std::size_t i=2;i<params.size();++i) {auto v=values(params[i]->data);x.core.insert(x.core.end(),v.begin(),v.end());}x.valid_lengths=valid;
    auto expected=ref::block_forward(x,reference(initial));auto out=linear(expected.output,wout,6,s.inner(),4);auto tape=layer.forward_owned(input.to(dev),initial,valid);close(tape->output(),out,"full projection forward");close_state(tape->snapshot_final_state(),expected.final_state);
    Tensor dy({2,3,4});fill(dy,.01f);for(int i=12;i<24;++i) dy.data()[i]=std::numeric_limits<float>::quiet_NaN();auto dyv=values(dy);for(int i=12;i<24;++i) dyv[i]=0;
    std::vector<double> dmixed(6*s.inner());for(int t=0;t<6;++t) for(int i=0;i<s.inner();++i) for(int d=0;d<4;++d) dmixed[t*s.inner()+i]+=dyv[t*4+d]*wout[d*s.inner()+i];
    auto g=ref::block_backward(x,reference(initial),dmixed,reference(seed));auto got=layer.backward_owned(tape,dy.to(dev),seed);close_state(got.initial_state,g.initial_state);
    std::vector<double> dx(24),dwin(win.size()),dwout(wout.size());
    for(int t=0;t<6;++t) for(int j=0;j<s.width();++j) for(int d=0;d<4;++d) {dx[t*4+d]+=g.projection[t*s.width()+j]*win[j*4+d];dwin[j*4+d]+=g.projection[t*s.width()+j]*input_values[t*4+d];}
    for(int t=0;t<6;++t) for(int d=0;d<4;++d) for(int i=0;i<s.inner();++i) dwout[d*s.inner()+i]+=dyv[t*4+d]*expected.output[t*s.inner()+i];
    close(got.input,dx,"input VJP");close(got.parameters[0],dwin,"in projection VJP");close(got.parameters[1],dwout,"out projection VJP");
    std::size_t off=0;for(std::size_t i=2;i<params.size();++i) {const auto n=std::size_t(params[i]->data.size);close(got.parameters[i],std::vector<double>(g.core.begin()+off,g.core.begin()+off+n),params[i]->name.c_str());off+=n;}
    layer.publish(got);for(auto* p:params) require(p->has_gradient(),"registry gradient not contributed");rejected([&]{layer.publish(got);},"duplicate publication");rejected([&]{tape->backward(dy.to(dev));},"duplicate backward");
    auto a=layer.forward_owned(input.to(dev),initial,valid),b=layer.forward_owned(input.to(dev),initial,valid);close(a->output(),b->output(),"independent tapes",0);a->cancel();rejected([&]{a->backward(dy.to(dev));},"cancelled backward");
    std::atomic<bool> thread_rejected{false};std::thread other([&]{try{b->output();}catch(...){thread_rejected=true;}});other.join();require(thread_rejected,"thread affinity");b->cancel();
}
void streaming(Device dev) {
    auto c=config(true,true,8,2);Mamba3Layer full(4,c),split(4,c);full.to(dev);split.to(dev);Tensor x({1,5,4});fill(x);Tensor dy({1,5,4});fill(dy,.01f);
    auto whole=full.forward_owned(x.to(dev));auto part1=split.forward_owned(x.slice(1,0,2).to(dev));auto part2=split.forward_owned(x.slice(1,2,5).to(dev),part1->snapshot_final_state());
    close(whole->output().slice(1,0,2),part1->output(),"stream prefix");close(whole->output().slice(1,2,5),part2->output(),"stream suffix");close_state(part2->snapshot_final_state(),reference(whole->snapshot_final_state()));
    auto gfull=full.backward_owned(whole,dy.to(dev));auto g2=split.backward_owned(part2,dy.slice(1,2,5).to(dev));auto g1=split.backward_owned(part1,dy.slice(1,0,2).to(dev),g2.initial_state);
    close(gfull.input.slice(1,0,2),g1.input,"stream dinput1");close(gfull.input.slice(1,2,5),g2.input,"stream dinput2");
    for(std::size_t i=0;i<gfull.parameters.size();++i) close(gfull.parameters[i],g1.parameters[i].add(g2.parameters[i]),"stream parameter adjoints");
    Mamba3Layer session(4,c);session.to(dev);session.set_training_mode(false);session.set_streaming_mode(true);session.forward(x.slice(1,0,2).to(dev));auto saved=session.snapshot_streaming_state(true);auto out=session.forward(x.slice(1,2,5).to(dev));session.restore_streaming_state(saved);close(out,session.forward(x.slice(1,2,5).to(dev)),"fork restore",0);
    auto batch=session.snapshot_streaming_state_batch(true);session.restore_streaming_state_batch({batch[0],batch[0]});require(session.streaming_batch_size()==2,"batch session restore");auto ckpt=std::filesystem::temp_directory_path()/"nsos-mamba3-layer-checkpoint.bin";session.save_checkpoint(ckpt.string(),true);Mamba3Layer restored(4,c);restored.to(dev);restored.load_checkpoint(ckpt.string(),true);require(restored.streaming_batch_size()==2,"checkpoint session batch");
    auto p=restored.parameters();p[4]->mark_updated();auto stale=restored.forward_owned(x.to(dev));auto grad=restored.backward_owned(stale,dy.to(dev));p[4]->mark_updated();rejected([&]{restored.publish(grad);},"stale weight publication");
    auto wrong=config(false,true,8);Mamba3Layer incompatible(4,wrong);rejected([&]{incompatible.load_checkpoint(ckpt.string());},"MIMO checkpoint mismatch");std::filesystem::remove(ckpt);
}
ModelConfig model_config(bool mimo) {ModelConfig c;c.architecture_schema_version=3;c.mamba3_enabled=true;c.mamba3_mimo=mimo;c.mamba3_mimo_rank=2;c.mamba3_outproj_norm=true;c.mamba3_state_dim=128;c.num_layers=1;c.d_model=8;c.vocab_size=32;c.n_heads=2;c.n_kv_heads=1;c.mamba_expand=1;c.mamba_head_dim=4;c.use_moe=false;c.use_ttt=false;c.attention_period=8;c.max_context_tokens=16;return c;}
void model_and_trainer(Device dev) {
    auto cfg=model_config(true);JambaModel model(cfg,dev);Context context;auto logits=model.forward_ids({1,2,3},&context);model.backward(Tensor::ones(logits.shape.dims,dev).mul(.001f),context);
    bool found=false;for(auto* p:model.parameters()) if(p->name.find("mamba3.")!=std::string::npos) {found=true;require(p->has_gradient(),"integrated parameter gradient");}require(found,"Mamba3 absent from registry");
    const auto exclusions=model.no_weight_decay_parameters();require(exclusions.size()==2,"Mamba3 optimizer exclusions missing");
    for(auto* p:exclusions) require(p->name.find("mamba3.dt_bias")!=std::string::npos||p->name.find("mamba3.D")!=std::string::npos,"unexpected optimizer exclusion");
    Trainer trainer(&model,.001f);trainer.weight_decay=0;trainer.phase_scheduler.progressive_qat_enabled=false;const auto before=values(model.parameters()[3]->data);float loss=trainer.train_step({1,2,3},{2,3,4});require(std::isfinite(loss),"integrated Trainer loss");bool updated=false;for(auto* p:model.parameters()) if(p->name.find("mamba3.in_proj.weight")!=std::string::npos) updated|=p->version>1;require(updated,"Trainer did not update Mamba3");
    auto path=std::filesystem::temp_directory_path()/"nsos-mamba3-integrated.bin";model.save(path.string());
    auto sidecar=std::filesystem::temp_directory_path()/"nsos-mamba3-integrated.trainer.bin";trainer.save_training_state(sidecar.string(),path.string());
    JambaModel restored(cfg,dev);restored.load(path.string());Trainer resumed(&restored,.01f);resumed.load_training_state(sidecar.string(),path.string());
    const float uninterrupted=trainer.train_step({1,2,3},{2,3,4}),continued=resumed.train_step({1,2,3},{2,3,4});require(std::abs(uninterrupted-continued)<1e-6f,"Trainer resumed loss mismatch");
    const auto current=model.parameters(),reloaded=restored.parameters();require(current.size()==reloaded.size(),"resumed registry");for(std::size_t i=0;i<current.size();++i) close(current[i]->data,reloaded[i]->data,"Trainer continuation parameter",2e-6);
    // Re-save the continued generation for subsequent architecture checks.
    model.save(path.string());restored.load(path.string());
    const auto saved=model.parameters(),loaded=restored.parameters();for(std::size_t i=0;i<saved.size();++i) close(saved[i]->data,loaded[i]->data,"checkpoint exact parameter roundtrip",0);
    model.set_training_mode(false);restored.set_training_mode(false);close(model.forward_ids({1,2,3}),restored.forward_ids({1,2,3}),"integrated checkpoint roundtrip",0);
    auto incompatible=cfg;incompatible.mamba3_mimo=false;JambaModel bad(incompatible,dev);rejected([&]{bad.load(path.string(),false);},"MIMO checkpoint strict=false mismatch");auto legacy=cfg;legacy.architecture_schema_version=2;legacy.mamba3_enabled=false;JambaModel baseline(legacy,dev);rejected([&]{baseline.load(path.string(),false);},"Mamba3/Mamba2 checkpoint mismatch");
    model.set_streaming_inference(true);auto pref=model.forward_ids({1,2});auto snap=model.fork_session(true);auto a=model.forward_ids({3});model.restore_session(snap);close(a,model.forward_ids({3}),"integrated session",0);require(model.runtime_telemetry().mamba3_layers.size()==1,"Mamba3 telemetry missing");std::filesystem::remove(path);std::filesystem::remove(sidecar);
}
void failures(Device dev) {
    auto c=config(true,true,8,2);Mamba3Layer layer(4,c);layer.to(dev);Tensor x({1,2,4});fill(x);Tensor dy({1,2,4});fill(dy,.01f);
    auto tape=layer.forward_owned(x.to(dev));Tensor bad_dy=dy.clone();bad_dy.data()[0]=std::numeric_limits<float>::quiet_NaN();auto bad=layer.backward_owned(tape,bad_dy.to(dev));require(tape->audit_status()[0]!=0,"nonfinite adjoint accepted");rejected([&]{layer.publish(bad);},"failed gradients published");for(auto* p:layer.parameters()) require(!p->has_gradient(),"failed VJP contributed gradients");
    auto cancelled=layer.forward_owned(x.to(dev));auto g=layer.backward_owned(cancelled,dy.to(dev));cancelled->cancel();rejected([&]{layer.publish(g);},"cancelled computed gradient published");
    rejected([&]{layer.forward_owned(x.to(dev),{},std::vector<int>{3});},"invalid prefix accepted");
    Tensor allbad=x.clone();allbad.data()[0]=std::numeric_limits<float>::infinity();auto broken=layer.forward_owned(allbad.to(dev));require(broken->audit_status()[0]!=0,"nonfinite input accepted");for(double y:values(broken->output())) require(y==0,"invalid output not gated");
    set_matmul_precision_mode(1);rejected([&]{layer.forward_owned(x.to(dev));},"mixed GEMM accepted as exact Mamba3");set_matmul_precision_mode(0);
}
void pack() {
    auto cfg=model_config(false);cfg.vocab_size=128;InferenceEngine engine;require(engine.load_model("",cfg),"SDK Mamba3 creation");const auto root=std::filesystem::temp_directory_path()/"nsos-mamba3-integral-pack";std::filesystem::create_directories(root);require(engine.save_model_pack(root.string()),"Mamba3 pack export");InferenceEngine restored;require(restored.load_model(root.string()),"Mamba3 pack import");auto got=restored.model_config();require(got.architecture_schema_version==3&&got.mamba3_enabled&&got.mamba3_state_dim==128&&got.mamba3_outproj_norm,"Mamba3 pack config lost");
    // Only explicitly created files are removed; do not recursively delete a
    // preexisting temp directory shared with another invocation.
}
}
int main(int argc,char** argv) {try {
    const Device dev=argc>1&&std::string(argv[1])=="--gpu"?Device::GPU:Device::CPU;
#ifdef USE_CUDA
    if(dev==Device::GPU) {int count=0;require(cudaGetDeviceCount(&count)==cudaSuccess&&count>0,"GPU required");require(cudaSetDevice(0)==cudaSuccess,"GPU selection");}
#endif
    set_matmul_precision_mode(0);for(bool mimo:{false,true}) for(bool norm:{false,true}) oracle(mimo,norm,dev);streaming(dev);failures(dev);model_and_trainer(dev);if(dev==Device::CPU) pack();
    std::cout<<"Mamba3 integral layer / Jamba / Trainer / checkpoint / sessions / pack PASS\n";return 0;
} catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
