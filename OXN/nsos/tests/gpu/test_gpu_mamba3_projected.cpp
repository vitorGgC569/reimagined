#include "gpu_parity_common.h"
#include "gpu_mamba3_projected.h"
#include "mamba3_preprocessing.h"
#include "cuda/device_buffer.h"
#include "gpu_execution.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

using namespace nsos;
namespace op=nsos::mamba3_projected;
namespace prep=nsos::mamba3_preprocessing;
namespace ref=nsos::mamba3_reference;
namespace pg=nsos::mamba3_preprocess_gpu;
namespace {
void require(bool v,const char* name) {if (!v) throw std::runtime_error(name);}
void fill(Tensor& t,float a,float f,float offset=0) {for (size_t i=0;i<t.size;++i) t.data()[i]=offset+a*std::sin(f*static_cast<float>(i+1));}
std::vector<double> values(const Tensor& t) {if (!t.size) return {};auto c=t.to(Device::CPU);return {c.data(),c.data()+c.size};}
void close(const Tensor& t,const std::vector<double>& expected,const std::string& name) {
    const auto got=values(t);require(got.size()==expected.size(),"adjoint extent mismatch");
    for (size_t i=0;i<got.size();++i) if (!std::isfinite(got[i])||!std::isfinite(expected[i])||std::abs(got[i]-expected[i])>3e-4+8e-4*std::max(std::abs(got[i]),std::abs(expected[i])))
        throw std::runtime_error(name+" at "+std::to_string(i)+" got="+std::to_string(got[i])+" expected="+std::to_string(expected[i]));
}
ref::State reference(const op::State& s) {return {values(s.phase),values(s.ssm),values(s.k),values(s.v)};}
void close(const op::State& a,const ref::State& b) {close(a.phase,b.phase,"phase state");close(a.ssm,b.ssm,"SSM state");close(a.k,b.k,"K state");close(a.v,b.v,"V state");}
std::vector<Tensor*> fields(op::Operands& x) {return {&x.q,&x.k,&x.v,&x.z,&x.raw_a,&x.raw_dt,&x.trap,&x.angles,&x.q_norm,&x.k_norm,&x.dt_bias,&x.q_bias,&x.k_bias,&x.d};}
op::Operands move(op::Operands x,Device dev) {for (auto* p:fields(x)) if (p->size) *p=p->to(dev);return x;}
op::State move(op::State s,Device dev) {return {s.phase.to(dev),s.ssm.to(dev),s.k.to(dev),s.v.to(dev)};}
op::State state(op::Shape s,bool seed=false) {
    op::State x{Tensor({s.batch,s.heads,s.rotary_pairs}),Tensor({s.batch,s.heads,s.head_dim,s.state_dim}),Tensor({s.batch,s.heads,s.state_dim}),Tensor({s.batch,s.heads,s.head_dim})};
    fill(x.phase,.02f,.29f,seed?0:.23f);fill(x.ssm,.009f,.011f);fill(x.k,.027f,.13f);fill(x.v,.023f,.17f);return x;
}
op::Operands fixture(op::Shape s,bool gate=true,bool skip=true) {
    op::Operands x{Tensor({s.batch,s.sequence,s.groups,s.state_dim}),Tensor({s.batch,s.sequence,s.groups,s.state_dim}),Tensor({s.batch,s.sequence,s.heads,s.head_dim}),
        gate?Tensor({s.batch,s.sequence,s.heads,s.head_dim}):Tensor(),Tensor({s.batch,s.sequence,s.heads}),Tensor({s.batch,s.sequence,s.heads}),Tensor({s.batch,s.sequence,s.heads}),
        Tensor({s.batch,s.sequence,s.rotary_pairs}),Tensor({s.state_dim}),Tensor({s.state_dim}),Tensor({s.heads}),Tensor({s.heads,s.state_dim}),Tensor({s.heads,s.state_dim}),skip?Tensor({s.heads}):Tensor()};
    fill(x.q,.12f,.021f);fill(x.k,.13f,.027f);fill(x.v,.11f,.041f);if (gate) fill(x.z,.36f,.017f);
    fill(x.raw_a,2,.077f,-.5f);fill(x.raw_dt,.8f,.037f,.4f);fill(x.trap,.4f,.049f);fill(x.angles,.09f,.053f,.4f);
    fill(x.q_norm,.07f,.071f,.9f);fill(x.k_norm,.06f,.091f,1.1f);fill(x.dt_bias,.2f,.031f,-3);
    fill(x.q_bias,.04f,.071f,.7f);fill(x.k_bias,.05f,.097f,.8f);if (skip) fill(x.d,.13f,.22f,.3f);return x;
}
prep::Inputs reference(op::Shape s,const op::Operands& x,const std::vector<int>& valid,op::Config c={}) {
    return {{s.batch,s.sequence,s.heads,s.groups,s.head_dim,s.state_dim,s.rotary_pairs},values(x.q),values(x.k),values(x.v),values(x.z),values(x.raw_a),values(x.raw_dt),values(x.trap),
        values(x.angles),values(x.q_norm),values(x.k_norm),values(x.dt_bias),values(x.q_bias),values(x.k_bias),values(x.d),valid,c.norm_eps,c.a_floor};
}
void close(op::Operands actual,const prep::Inputs& x) {
    const std::vector<const std::vector<double>*> expected{&x.q,&x.k,&x.v,&x.z,&x.raw_a,&x.raw_dt,&x.trap,&x.angles,&x.q_norm,&x.k_norm,&x.dt_bias,&x.q_bias,&x.k_bias,&x.d};
    const auto all=fields(actual);for (size_t i=0;i<all.size();++i) close(*all[i],*expected[i],"field"+std::to_string(i));
}
void poison(op::Shape s,op::Operands& x,Tensor& dy,const std::vector<int>& valid) {
    const auto all=fields(x);
    for (int b=0;b<s.batch;++b) for (int t=valid[b];t<s.sequence;++t) {
        for (int i=0;i<8;++i) if (all[i]->size) {
            const auto width=all[i]->size/(static_cast<size_t>(s.batch)*s.sequence);
            std::fill_n(all[i]->data()+(static_cast<size_t>(b)*s.sequence+t)*width,width,std::numeric_limits<float>::quiet_NaN());
        }
        std::fill_n(dy.data()+(static_cast<size_t>(b)*s.sequence+t)*s.heads*s.head_dim,s.heads*s.head_dim,std::numeric_limits<float>::quiet_NaN());
    }
}
template<class F> void rejected(F f,const char* name) {bool bad=false;try {f();} catch (const std::exception&) {bad=true;}require(bad,name);}
void parity(op::Shape s,op::Config c={},bool gate=true,bool skip=true) {
    auto x=fixture(s,gate,skip);auto initial=state(s),seed=state(s,true);Tensor dy(x.v.shape.dims);fill(dy,.12f,.031f);
    std::vector<int> valid(s.batch,s.sequence);if (s.batch>1) valid[0]=std::max(0,s.sequence-7);if (s.batch>2) valid.back()=0;
    poison(s,x,dy,valid);
    const auto rx=reference(s,x,valid,c);
    const auto ri=reference(initial),rs=reference(seed);
    const auto prepared=prep::forward(rx);
    const auto expected=ref::forward(prepared,ri);const auto adj=ref::backward(prepared,ri,values(dy),rs);const auto dx=prep::backward(rx,adj.input);
    const auto gx=move(x,Device::GPU);
    const auto gi=move(initial,Device::GPU),gs=move(seed,Device::GPU);const auto gdy=dy.to(Device::GPU);
    reset_gpu_transfer_stats();
    auto tape=op::Tape::forward(s,gx,gi,valid,c);auto grad=tape->backward(gdy,gs);
    const auto transfers=gpu_transfer_stats();require(transfers.d2h_calls==0&&transfers.device_synchronizations==0&&transfers.stream_synchronizations==0,"composed core performed host readback/explicit fence");
    require(tape->audit_status()==std::vector<int>(s.batch,0),"valid composed operands rejected");
    close(tape->output(),expected.output,"composed output");close(tape->snapshot_final_state(),expected.final_state);close(grad.input,dx);close(grad.initial_state,adj.initial_state);
    require(tape->preprocessing_workspace_bytes()==(static_cast<size_t>(s.batch)*s.sequence*s.groups*2+pg::partial_elements(s))*4,"preprocessing scratch size");
    rejected([&]{tape->backward(gdy);},"consumed composed tape reused");
    auto repeat=op::Tape::forward(s,gx,gi,valid,c);auto again=repeat->backward(gdy,gs);require(repeat->audit_status()==std::vector<int>(s.batch,0),"repeat status");
    auto a=fields(grad.input),b=fields(again.input);for (size_t i=0;i<a.size();++i) require(values(*a[i])==values(*b[i]),"composed gradients nondeterministic");
    require(values(grad.initial_state.phase)==values(again.initial_state.phase)&&values(grad.initial_state.ssm)==values(again.initial_state.ssm)&&
        values(grad.initial_state.k)==values(again.initial_state.k)&&values(grad.initial_state.v)==values(again.initial_state.v),"state adjoints nondeterministic");
    std::cout<<"Projected Mamba3 B/S/H/G/P/N/R/chunk="<<s.batch<<'/'<<s.sequence<<'/'<<s.heads<<'/'<<s.groups<<'/'<<s.head_dim<<'/'<<s.state_dim<<'/'<<s.rotary_pairs<<'/'<<s.chunk<<" preprocess_bytes="<<tape->preprocessing_workspace_bytes()<<'\n';
}
Tensor slice(const Tensor& t,int batch,int seq,int start,int count) {
    if (!t.size) return {};auto dims=t.shape.dims;dims[1]=count;Tensor out(dims);
    const auto src=t.to(Device::CPU);const auto width=t.size/(static_cast<size_t>(batch)*seq);
    for (int b=0;b<batch;++b) std::copy_n(src.data()+(static_cast<size_t>(b)*seq+start)*width,count*width,out.data()+static_cast<size_t>(b)*count*width);return out;
}
op::Operands slice(op::Operands x,int batch,int seq,int start,int count) {auto all=fields(x);for (int i=0;i<8;++i) *all[i]=slice(*all[i],batch,seq,start,count);return x;}
void streaming_ownership() {
    op::Shape s{2,35,4,2,7,6,3,16};const auto x=fixture(s);
    const auto initial=state(s),seed=state(s,true);Tensor dy(x.v.shape.dims);fill(dy,.1f,.031f);
    auto gx=move(x,Device::GPU);const auto gi=move(initial,Device::GPU),gs=move(seed,Device::GPU);const auto gdy=dy.to(Device::GPU);
    auto whole=op::Tape::forward(s,gx,gi,{22,35});auto dw=whole->backward(gdy,gs);
    auto a=s;a.sequence=16;auto b=s;b.sequence=19;
    auto head=op::Tape::forward(a,move(slice(x,2,35,0,16),Device::GPU),gi,{16,16});
    auto tail=op::Tape::forward(b,move(slice(x,2,35,16,19),Device::GPU),head->snapshot_final_state(),{6,19});
    auto db=tail->backward(slice(dy,2,35,16,19).to(Device::GPU),gs);auto da=head->backward(slice(dy,2,35,0,16).to(Device::GPU),db.initial_state);
    require(whole->audit_status()==std::vector<int>({0,0})&&head->audit_status()==std::vector<int>({0,0})&&tail->audit_status()==std::vector<int>({0,0}),"streaming composition status");
    close(head->output(),values(slice(whole->output(),2,35,0,16)),"stream first");close(tail->output(),values(slice(whole->output(),2,35,16,19)),"stream last");
    close(tail->snapshot_final_state(),reference(whole->snapshot_final_state()));close(da.initial_state,reference(dw.initial_state));
    auto want_a=slice(dw.input,2,35,0,16),want_b=slice(dw.input,2,35,16,19);auto ga=fields(da.input),gb=fields(db.input),wa=fields(want_a),wb=fields(want_b),w=fields(dw.input);
    for (int i=0;i<8;++i) {close(*ga[i],values(*wa[i]),"stream adjoint first");close(*gb[i],values(*wb[i]),"stream adjoint last");}
    for (int i=8;i<14;++i) {auto sum=values(*ga[i]);const auto rhs=values(*gb[i]);for (size_t j=0;j<sum.size();++j) sum[j]+=rhs[j];close(*w[i],sum,"stream parameter sum");}
    auto immutable=op::Tape::forward(s,gx,gi,{22,35});Tensor replacement(x.raw_dt.shape.dims);fill(replacement,.01f,.1f,9);
    copy_tensor_bytes(gx.raw_dt.raw_data(),Device::GPU,replacement.data(),Device::CPU,replacement.size*4);
    auto owned=immutable->backward(gdy,gs);require(immutable->audit_status()==std::vector<int>({0,0}),"immutable source status");close(owned.input.raw_dt,values(dw.input.raw_dt),"owned raw DT");close(owned.input.q_norm,values(dw.input.q_norm),"owned norm weight");
    auto cancel=op::Tape::forward(s,move(x,Device::GPU));cancel->cancel();rejected([&]{cancel->backward(gdy);},"cancelled tape reused");
    bool foreign=false;std::thread other([&]{try {cancel->output();} catch (const std::exception&) {foreign=true;}});other.join();require(foreign,"foreign thread tape accepted");
    gpu::ExecutionContext lane;{gpu::ExecutionContext::Scope scope(lane);rejected([&]{cancel->audit_status();},"foreign stream tape accepted");}
    auto bad=move(x,Device::GPU);bad.q.shape.strides.back()=2;rejected([&]{op::Tape::forward(s,bad);},"bad stride accepted");
    rejected([&]{op::Tape::forward(s,move(x,Device::GPU),{}, {},{0,1e-4f});},"invalid eps accepted");
    rejected([&]{op::Tape::forward(s,move(x,Device::GPU),{}, {-1,35});},"invalid prefix accepted");
}
void streaming_state_closure() {
    // Legal projections produce previous K>64 and SSM>65536. Their snapshots
    // must remain legal entering states, not be rejected by a smaller bound.
    op::Shape s{1,2,1,1,1,2,1,16};auto x=fixture(s,false,false);
    std::fill_n(x.q.data(),x.q.size,1);std::fill_n(x.k.data(),x.k.size,1);
    std::fill_n(x.q_norm.data(),x.q_norm.size,1);std::fill_n(x.k_norm.data(),x.k_norm.size,64);
    std::fill_n(x.q_bias.data(),x.q_bias.size,0);std::fill_n(x.k_bias.data(),x.k_bias.size,64);
    std::fill_n(x.v.data(),x.v.size,64);std::fill_n(x.raw_a.data(),x.raw_a.size,-64);
    std::fill_n(x.raw_dt.data(),x.raw_dt.size,16);std::fill_n(x.dt_bias.data(),x.dt_bias.size,0);
    std::fill_n(x.trap.data(),x.trap.size,0);x.angles.data()[0]=std::atanh(1.0f/64);x.angles.data()[1]=-x.angles.data()[0];
    auto whole=op::Tape::forward(s,move(x,Device::GPU));require(whole->audit_status()==std::vector<int>({0}),"legal whole forward rejected");
    auto one=s;one.sequence=1;auto first=op::Tape::forward(one,move(slice(x,1,2,0,1),Device::GPU));const auto carry=first->snapshot_final_state();
    require(first->audit_status()==std::vector<int>({0}),"legal first chunk rejected");
    double max_k=0,max_h=0;for (double v:values(carry.k)) max_k=std::max(max_k,std::abs(v));for (double v:values(carry.ssm)) max_h=std::max(max_h,std::abs(v));
    require(max_k>64&&max_h>65536,"state closure fixture failed to cross old bounds");
    auto last=op::Tape::forward(one,move(slice(x,1,2,1,1),Device::GPU),carry);
    require(last->audit_status()==std::vector<int>({0}),"SISO rejects its own finite final state on next streaming call");
    Tensor dy(x.v.shape.dims);std::fill_n(dy.data(),dy.size,100); // finite adjoints are not bounded like projections
    auto seed=state(s,true);std::fill_n(seed.phase.data(),seed.phase.size,1000);std::fill_n(seed.ssm.data(),seed.ssm.size,77);
    std::fill_n(seed.k.data(),seed.k.size,100);std::fill_n(seed.v.data(),seed.v.size,250);
    auto dw=whole->backward(dy.to(Device::GPU),move(seed,Device::GPU));auto db=last->backward(slice(dy,1,2,1,1).to(Device::GPU),move(seed,Device::GPU));
    auto da=first->backward(slice(dy,1,2,0,1).to(Device::GPU),db.initial_state);
    require(whole->audit_status()==std::vector<int>({0})&&last->audit_status()==std::vector<int>({0})&&first->audit_status()==std::vector<int>({0}),"finite cross-call state adjoint rejected");
    const auto rx=reference(s,x,{});const auto rr=ref::forward(prep::forward(rx));const auto rg=ref::backward(prep::forward(rx),{},values(dy),reference(seed));
    close(whole->output(),rr.output,"large finite state primal");close(dw.input,prep::backward(rx,rg.input));close(dw.initial_state,rg.initial_state);
    close(last->snapshot_final_state(),reference(whole->snapshot_final_state()));close(da.initial_state,reference(dw.initial_state));
    auto a=slice(dw.input,1,2,0,1),b=slice(dw.input,1,2,1,1);auto ga=fields(da.input),gb=fields(db.input),wa=fields(a),wb=fields(b),w=fields(dw.input);
    for (int i=0;i<8;++i) {close(*ga[i],values(*wa[i]),"large state first VJP");close(*gb[i],values(*wb[i]),"large state last VJP");}
    for (int i=8;i<14;++i) {auto sum=values(*ga[i]);const auto rhs=values(*gb[i]);for (size_t j=0;j<sum.size();++j) sum[j]+=rhs[j];close(*w[i],sum,"large state parameter sum");}
    std::cout<<"State closure previousK="<<max_k<<" SSM="<<max_h<<" finite dy=100 and state seeds>64 PASS\n";
}
double dot(const std::vector<double>& a,const std::vector<double>& b) {require(a.size()==b.size(),"dot shape");double value=0;for (size_t i=0;i<a.size();++i) value+=a[i]*b[i];return value;}
double objective(const prep::Inputs& x,const ref::State& initial,const ref::State& seed,const std::vector<double>& dy) {
    const auto y=ref::forward(prep::forward(x),initial);return dot(y.output,dy)+dot(y.final_state.phase,seed.phase)+dot(y.final_state.ssm,seed.ssm)+dot(y.final_state.k,seed.k)+dot(y.final_state.v,seed.v);
}
void finite_differences() {
    op::Shape s{1,19,4,2,7,6,3,16};auto x=fixture(s);const auto initial=state(s),seed=state(s,true);Tensor dy(x.v.shape.dims);fill(dy,.13f,.021f);
    auto rx=reference(s,x,{});auto ri=reference(initial);const auto rs=reference(seed);
    const auto d=values(dy);
    auto tape=op::Tape::forward(s,move(x,Device::GPU),move(initial,Device::GPU));const auto grad=tape->backward(dy.to(Device::GPU),move(seed,Device::GPU));require(tape->audit_status()==std::vector<int>({0}),"FD composition status");
    auto check=[&](std::vector<double>& input,const Tensor& output,size_t at) {
        const double old=input[at],eps=1e-5;input[at]=old+eps;const double plus=objective(rx,ri,rs,d);input[at]=old-eps;const double minus=objective(rx,ri,rs,d);input[at]=old;
        const double fd=(plus-minus)/(2*eps),actual=values(output)[at];require(std::abs(actual-fd)<=3e-4+8e-4*std::abs(fd),"composed finite difference failed");
    };
    std::vector<std::vector<double>*> raw{&rx.q,&rx.k,&rx.v,&rx.z,&rx.raw_a,&rx.raw_dt,&rx.trap,&rx.angles,&rx.q_norm,&rx.k_norm,&rx.dt_bias,&rx.q_bias,&rx.k_bias,&rx.d};
    auto gradients=grad.input;auto all=fields(gradients);for (size_t i=0;i<raw.size();++i) check(*raw[i],*all[i],std::min(size_t(11),raw[i]->size()-1));
    check(ri.phase,grad.initial_state.phase,3);check(ri.ssm,grad.initial_state.ssm,17);check(ri.k,grad.initial_state.k,5);check(ri.v,grad.initial_state.v,7);
    // Exactly on the declared floor chooses zero, not central FD at a kink.
    x.raw_a.data()[0]=-1;auto floor=op::Tape::forward(s,move(x,Device::GPU),{}, {},{1e-5f,.5f});auto fg=floor->backward(dy.to(Device::GPU));
    require(floor->audit_status()==std::vector<int>({0}),"floor equality status");require(values(fg.input.raw_a)[0]==0,"floor equality subgradient not zero");
}
void failure_gates() {
    op::Shape s{2,17,4,2,7,6,3,16};auto x=fixture(s);Tensor dy(x.v.shape.dims);fill(dy,.1f,.019f);
    for (int kind=0;kind<4;++kind) {
        auto raw=fixture(s);
        if (kind==0) raw.q.data()[0]=std::numeric_limits<float>::quiet_NaN();
        else if (kind==1) raw.raw_dt.data()[0]=64; // softplus(raw+bias)>16
        else if (kind==2) raw.v.data()[0]=std::numeric_limits<float>::quiet_NaN(); // downstream-only failure
        else raw.angles.data()[0]=65;
        auto t=op::Tape::forward(s,move(raw,Device::GPU));auto g=t->backward(dy.to(Device::GPU));require(t->audit_status()==std::vector<int>({2,0}),"upstream/downstream failure was reset");
        const auto out=values(t->output());for (size_t i=0;i<out.size()/2;++i) require(out[i]==0,"failed composed output leaked");
        auto all=fields(g.input);
        for (int i=0;i<8;++i) {const auto v=values(*all[i]);for (size_t j=0;j<v.size()/2;++j) require(v[j]==0,"invalid batch gradient leaked");}
        for (int i=8;i<14;++i) for (double v:values(*all[i])) require(v==0,"invalid op published shared gradient");
        const auto ds=values(g.initial_state.ssm);for (size_t i=0;i<ds.size()/2;++i) require(ds[i]==0,"invalid initial adjoint leaked");
    }
    std::vector<int> zero{0,0};poison(s,x,dy,zero);x.q_norm.data()[0]=std::numeric_limits<float>::quiet_NaN();x.dt_bias.data()[0]=std::numeric_limits<float>::quiet_NaN();
    const auto initial=state(s),seed=state(s,true);auto empty=op::Tape::forward(s,move(x,Device::GPU),move(initial,Device::GPU),zero);auto g=empty->backward(dy.to(Device::GPU),move(seed,Device::GPU));
    require(empty->audit_status()==std::vector<int>({0,0}),"zero-length data/parameters read");close(empty->snapshot_final_state(),reference(initial));close(g.initial_state,reference(seed));
    for (auto* p:fields(g.input)) for (double v:values(*p)) require(v==0,"zero-length gradient nonzero");
}
void raw_bounds() {
#ifdef USE_CUDA
    op::Shape s{2,33,4,2,7,6,3,16};auto x=move(fixture(s),Device::GPU);
    cuda_detail::DeviceBuffer<int> valid,status;require(valid.ensure(s.batch+4)&&status.ensure(s.batch+4),"raw metadata allocation");
    int lengths[6]={33,0,9000,9000,9000,9000},issues[6]={0,0,9000,9000,9000,9000};
    require(cudaMemcpy(valid.get(),lengths,sizeof(lengths),cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(status.get(),issues,sizeof(issues),cudaMemcpyHostToDevice)==cudaSuccess,"raw metadata upload");
    auto guarded=[](size_t n) {Tensor c({static_cast<int>(n+4)});std::fill_n(c.data(),c.size,123456.0f);return c.to(Device::GPU);};
    Tensor q=guarded(x.q.size),k=guarded(x.k.size),adt=guarded(x.raw_a.size),dt=guarded(x.raw_dt.size),angles=guarded(static_cast<size_t>(s.batch)*s.sequence*s.heads*s.rotary_pairs),inv=guarded(static_cast<size_t>(s.batch)*s.sequence*s.groups*2);
    pg::Input input{x.q.raw_data(),x.k.raw_data(),x.raw_a.raw_data(),x.raw_dt.raw_data(),x.angles.raw_data(),x.q_norm.raw_data(),x.k_norm.raw_data(),x.dt_bias.raw_data(),valid.get(),inv.raw_data()};
    pg::Prepared out{q.raw_data(),k.raw_data(),adt.raw_data(),dt.raw_data(),angles.raw_data(),inv.raw_data()};
    auto alias=out;alias.k=alias.q;require(!pg::forward(s,{},input,alias,status.get()),"aliased preprocessing output accepted");require(pg::forward(s,{},input,out,status.get()),"raw preprocessing forward enqueue");
    Tensor dq=guarded(x.q.size),dk=guarded(x.k.size),da=guarded(x.raw_a.size),ddt=guarded(x.raw_dt.size),dangles=guarded(x.angles.size),dqn=guarded(x.q_norm.size),dkn=guarded(x.k_norm.size),db=guarded(x.dt_bias.size),partials=guarded(pg::partial_elements(s));
    Tensor gq(x.q.shape.dims),gk(x.k.shape.dims),ga(x.raw_a.shape.dims),gd(x.raw_dt.shape.dims),gang({s.batch,s.sequence,s.heads,s.rotary_pairs});
    fill(gq,.11f,.031f);fill(gk,.12f,.021f);fill(ga,.13f,.017f);fill(gd,.14f,.019f);fill(gang,.15f,.027f);
    const auto ugq=gq.to(Device::GPU),ugk=gk.to(Device::GPU),uga=ga.to(Device::GPU),ugd=gd.to(Device::GPU),uang=gang.to(Device::GPU);
    pg::Adjoint grad{ugq.raw_data(),ugk.raw_data(),uga.raw_data(),ugd.raw_data(),uang.raw_data()};
    pg::Gradient dest{dq.raw_data(),dk.raw_data(),da.raw_data(),ddt.raw_data(),dangles.raw_data(),dqn.raw_data(),dkn.raw_data(),db.raw_data()};
    auto wrong=dest;wrong.k=wrong.q;require(!pg::backward(s,{},input,grad,wrong,partials.raw_data(),status.get()),"aliased preprocessing gradients accepted");
    require(pg::backward(s,{},input,grad,dest,partials.raw_data(),status.get()),"raw preprocessing backward enqueue");nsos::gpu_parity_test::cuda_sync_or_throw("raw preprocessing canaries");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"raw status download");require(issues[0]==0&&issues[1]==0,"raw preprocessing status");for (int i=2;i<6;++i) require(issues[i]==9000,"status canary overwritten");
    for (auto* t:{&q,&k,&adt,&dt,&angles,&inv,&dq,&dk,&da,&ddt,&dangles,&dqn,&dkn,&db,&partials}) {const auto v=values(*t);for (size_t i=v.size()-4;i<v.size();++i) require(v[i]==123456.0,"preprocessing capacity overrun");}
    const float nan=std::numeric_limits<float>::quiet_NaN();require(cudaMemcpy(inv.raw_data(),&nan,sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess,"inverse poison upload");
    require(pg::backward(s,{},input,grad,dest,partials.raw_data(),status.get()),"raw invalid-inverse enqueue");nsos::gpu_parity_test::cuda_sync_or_throw("invalid inverse gate");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"inverse status download");require(issues[0]==2&&issues[1]==0,"invalid inverse not rejected");
    const auto bad_norm=values(dqn);for (int n=0;n<s.state_dim;++n) require(bad_norm[n]==0,"invalid inverse published norm gradient");
    // Finite adjoints can still overflow during RMS VJP. Gate computed
    // failures as status3, without mistaking them for input range failure.
    require(pg::forward(s,{},input,out,status.get()),"raw inverse restore enqueue");
    Tensor giant(x.q.shape.dims);std::fill_n(giant.data(),giant.size,2e38f);const auto huge=giant.to(Device::GPU);
    auto overflow=grad;overflow.q=huge.raw_data();
    require(pg::backward(s,{},input,overflow,dest,partials.raw_data(),status.get()),"raw overflow enqueue");nsos::gpu_parity_test::cuda_sync_or_throw("raw numerical overflow gate");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"overflow status download");require(issues[0]==3,"finite VJP overflow not latched");
    Tensor rv=guarded(x.v.size),rz=guarded(x.z.size),rt=guarded(x.trap.size),rqb=guarded(x.q_bias.size),rkb=guarded(x.k_bias.size),rd=guarded(x.d.size);
    const size_t bh=static_cast<size_t>(s.batch)*s.heads;
    Tensor diph=guarded(bh*s.rotary_pairs),dih=guarded(bh*s.head_dim*s.state_dim),dik=guarded(bh*s.state_dim),div=guarded(bh*s.head_dim);
    mamba3_siso_gpu::Gradient chain{nullptr,nullptr,rv.raw_data(),rz.raw_data(),nullptr,nullptr,rt.raw_data(),nullptr,rqb.raw_data(),rkb.raw_data(),rd.raw_data()};
    mamba3_siso_gpu::State di{diph.raw_data(),dih.raw_data(),dik.raw_data(),div.raw_data()};
    auto bad_chain=chain;bad_chain.q_bias=dest.q_norm;
    require(!pg::gate_chain(s,status.get(),dest,bad_chain,di),"aliased chain gradient gate accepted");
    require(pg::gate_chain(s,status.get(),dest,chain,di),"chain gate enqueue");nsos::gpu_parity_test::cuda_sync_or_throw("composed numerical gradient gate");
    for (auto* t:{&rv,&rz,&rt,&diph,&dih,&dik,&div}) {
        const auto v=values(*t);for (size_t i=0;i<(v.size()-4)/2;++i) require(v[i]==0,"chain failure leaked initial/identity adjoint");
        for (size_t i=v.size()-4;i<v.size();++i) require(v[i]==123456.0,"chain gate capacity overrun");
    }
    for (auto* t:{&rqb,&rkb,&rd}) {const auto v=values(*t);for (size_t i=0;i<v.size()-4;++i) require(v[i]==0,"chain failure published shared adjoint");}

    // Preserve preprocessor status3 in the SISO forward itself; no D2H is
    // needed to decide whether a recurrence is allowed to consume the batch.
    issues[0]=3;issues[1]=0;require(cudaMemcpy(status.get(),issues,sizeof(issues),cudaMemcpyHostToDevice)==cudaSuccess,"upstream status upload");
    cuda_detail::DeviceBuffer<int> rec_status;require(rec_status.ensure(s.batch),"recurrence status allocation");
    Tensor rec_out=guarded(x.v.size),boundaries=guarded(mamba3_siso_gpu::boundary_elements(s));
    mamba3_siso_gpu::Input rx{out.q,out.k,x.v.raw_data(),x.z.raw_data(),out.adt,out.dt,x.trap.raw_data(),out.angles,x.q_bias.raw_data(),x.k_bias.raw_data(),x.d.raw_data(),valid.get()};
    require(!mamba3_siso_gpu::forward(s,rx,{},di,rec_out.raw_data(),boundaries.raw_data(),status.get(),status.get()),"aliased upstream status accepted");
    require(mamba3_siso_gpu::forward(s,rx,{},di,rec_out.raw_data(),boundaries.raw_data(),rec_status.get(),status.get()),"inherited recurrence gate enqueue");
    nsos::gpu_parity_test::cuda_sync_or_throw("inherited recurrence status");
    require(cudaMemcpy(issues,rec_status.get(),s.batch*sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess,"inherited recurrence status download");require(issues[0]==3&&issues[1]==0,"SISO reset inherited failure");
    lengths[0]=-1;lengths[1]=34;require(cudaMemcpy(valid.get(),lengths,sizeof(lengths),cudaMemcpyHostToDevice)==cudaSuccess,"bad prefix upload");
    require(pg::forward(s,{},input,out,status.get()),"raw bad-prefix enqueue");nsos::gpu_parity_test::cuda_sync_or_throw("raw bad-prefix gate");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"prefix status download");require(issues[0]==1&&issues[1]==1,"raw invalid prefix missed");
    require(cudaMemcpy(lengths,valid.get(),sizeof(lengths),cudaMemcpyDeviceToHost)==cudaSuccess,"valid canary download");for (int i=2;i<6;++i) require(lengths[i]==9000,"prefix canary overwritten");
#endif
}
}
int main() {
    return nsos::gpu_parity_test::run_parity("mamba3_projected",[] {
        parity({3,1,4,2,3,4,1,16});parity({2,17,4,1,7,6,1,16});parity({2,33,4,2,16,32,8,32});
        parity({1,65,2,2,64,64,32,16},{},false,false);parity({2,129,4,4,7,6,3,32});parity({1,257,4,2,16,32,16,32});
        parity({2,65,4,2,7,6,3,16},{.002f,.75f});parity({1,1025,2,1,3,4,2,32});
        {gpu::ExecutionContext lane;gpu::ExecutionContext::Scope scope(lane);parity({2,35,4,2,7,6,3,16});}
        streaming_ownership();streaming_state_closure();finite_differences();failure_gates();raw_bounds();
    });
}
