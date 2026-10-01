#include "gpu_parity_common.h"
#include "gpu_mamba3_siso.h"
#include "mamba3_reference.h"
#include "gpu_execution.h"
#include "cuda/device_buffer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

using namespace nsos;
namespace op=nsos::mamba3_siso;
namespace ref=nsos::mamba3_reference;
namespace mg=nsos::mamba3_siso_gpu;
namespace {
void require(bool ok,const char* name) {if (!ok) throw std::runtime_error(name);}
void fill(Tensor& t,float amplitude,float frequency,float offset=0) {
    for (size_t i=0;i<t.size;++i) t.data()[i]=offset+amplitude*std::sin(frequency*static_cast<float>(i+1));
}
std::vector<double> values(const Tensor& t) {
    if (!t.size) return {};auto cpu=t.to(Device::CPU);return {cpu.data(),cpu.data()+cpu.size};
}
Tensor device(const std::vector<double>& v,const std::vector<int>& dims) {
    Tensor t(dims,Device::CPU);require(t.size==v.size(),"reference upload shape");
    for (size_t i=0;i<v.size();++i) t.data()[i]=static_cast<float>(v[i]);return t.to(Device::GPU);
}
op::Operands move(const op::Operands& x,Device dev) {
    auto cv=[&](const Tensor& t){return t.size?t.to(dev):Tensor();};
    return {cv(x.q),cv(x.k),cv(x.v),cv(x.z),cv(x.adt),cv(x.dt),cv(x.trap),cv(x.angles),cv(x.q_bias),cv(x.k_bias),cv(x.d)};
}
op::State move(const op::State& x,Device dev) {return {x.phase.to(dev),x.ssm.to(dev),x.k.to(dev),x.v.to(dev)};}
ref::Geometry geometry(op::Shape s) {return {s.batch,s.sequence,s.heads,s.groups,s.head_dim,s.state_dim,s.rotary_pairs};}
ref::Inputs reference(const op::Shape& s,const op::Operands& x,const std::vector<int>& valid) {
    return {geometry(s),values(x.q),values(x.k),values(x.v),values(x.z),values(x.adt),values(x.dt),values(x.trap),
        values(x.angles),values(x.q_bias),values(x.k_bias),values(x.d),valid};
}
ref::State reference(const op::State& x) {return {values(x.phase),values(x.ssm),values(x.k),values(x.v)};}
op::State state(const op::Shape& s,bool seed=false) {
    op::State x{Tensor({s.batch,s.heads,s.rotary_pairs}),Tensor({s.batch,s.heads,s.head_dim,s.state_dim}),
        Tensor({s.batch,s.heads,s.state_dim}),Tensor({s.batch,s.heads,s.head_dim})};
    fill(x.phase,seed?.021f:.05f,.27f,seed?0:.23f);fill(x.ssm,seed?.017f:.009f,.013f);
    fill(x.k,seed?.023f:.025f,.19f);fill(x.v,seed?.029f:.03f,.11f);return x;
}
op::Operands operands(const op::Shape& s,bool gate=true,bool skip=true) {
    op::Operands x{Tensor({s.batch,s.sequence,s.groups,s.state_dim}),Tensor({s.batch,s.sequence,s.groups,s.state_dim}),
        Tensor({s.batch,s.sequence,s.heads,s.head_dim}),gate?Tensor({s.batch,s.sequence,s.heads,s.head_dim}):Tensor(),
        Tensor({s.batch,s.sequence,s.heads}),Tensor({s.batch,s.sequence,s.heads}),Tensor({s.batch,s.sequence,s.heads}),
        Tensor({s.batch,s.sequence,s.heads,s.rotary_pairs}),Tensor({s.heads,s.state_dim}),Tensor({s.heads,s.state_dim}),skip?Tensor({s.heads}):Tensor()};
    fill(x.q,.18f,.019f);fill(x.k,.16f,.029f);fill(x.v,.12f,.043f);if (gate) fill(x.z,.37f,.017f);
    fill(x.adt,.013f,.079f,-.042f);fill(x.dt,.008f,.051f,.037f);fill(x.trap,.4f,.039f);fill(x.angles,.09f,.023f);
    fill(x.q_bias,.05f,.071f,.71f);fill(x.k_bias,.04f,.097f,.83f);if (skip) fill(x.d,.14f,.22f,.3f);return x;
}
void close(const Tensor& t,const std::vector<double>& expected,const std::string& name,double atol=3e-4,double rtol=8e-4) {
    const auto actual=values(t);require(actual.size()==expected.size(),"GPU/reference extent mismatch");
    for (size_t i=0;i<actual.size();++i) if (!std::isfinite(actual[i])||!std::isfinite(expected[i])||std::abs(actual[i]-expected[i])>atol+rtol*std::max(std::abs(actual[i]),std::abs(expected[i])))
        throw std::runtime_error(name+" at "+std::to_string(i)+" GPU="+std::to_string(actual[i])+" ref="+std::to_string(expected[i]));
}
void close(const op::State& actual,const ref::State& expected,const char* name) {
    close(actual.phase,expected.phase,std::string(name)+" phase");close(actual.ssm,expected.ssm,std::string(name)+" ssm");
    close(actual.k,expected.k,std::string(name)+" K");close(actual.v,expected.v,std::string(name)+" V");
}
void close(const op::Operands& actual,const ref::Inputs& expected) {
    close(actual.q,expected.q,"dQ");close(actual.k,expected.k,"dK");close(actual.v,expected.v,"dV");close(actual.z,expected.z,"dZ");
    close(actual.adt,expected.adt,"dADT");close(actual.dt,expected.dt,"dDT");close(actual.trap,expected.trap,"dTrap");close(actual.angles,expected.angles,"dAngles");
    close(actual.q_bias,expected.q_bias,"dQBias");close(actual.k_bias,expected.k_bias,"dKBias");close(actual.d,expected.d,"dD");
}
template<class F> void rejected(F fn,const char* name) {bool threw=false;try {fn();} catch (const std::exception&) {threw=true;} require(threw,name);}
void poison(op::Operands& x,Tensor& dy,const op::Shape& s,const std::vector<int>& valid) {
    for (int b=0;b<s.batch;++b) for (int t=valid[b];t<s.sequence;++t) {
        for (auto pair:{std::pair{&x.q,s.groups*s.state_dim},std::pair{&x.k,s.groups*s.state_dim},std::pair{&x.v,s.heads*s.head_dim},std::pair{&x.z,s.heads*s.head_dim},
            std::pair{&x.adt,s.heads},std::pair{&x.dt,s.heads},std::pair{&x.trap,s.heads},std::pair{&x.angles,s.heads*s.rotary_pairs},std::pair{&dy,s.heads*s.head_dim}})
            if (pair.first->size) std::fill_n(pair.first->data()+(static_cast<size_t>(b)*s.sequence+t)*pair.second,pair.second,std::numeric_limits<float>::quiet_NaN());
    }
}
void parity(op::Shape s,bool gate=true,bool skip=true,bool phase_wrap=false) {
    require(mg::supported(s),"Mamba3 GPU capability not supported (no successful skip)");
    auto x=operands(s,gate,skip);auto initial=state(s),seed=state(s,true);
    if (phase_wrap) {
        // Multiple modulo crossings on both sides of a replay boundary. Small
        // oscillatory test angles alone would not exercise this state contract.
        fill(x.angles,.07f,.023f,.6f);fill(x.dt,.04f,.051f,1.25f);
        fill(x.adt,.01f,.079f,-.35f);fill(initial.phase,.02f,.27f,6.23f);
        for (size_t i=0;i<x.angles.size;++i)
            if ((i/s.rotary_pairs)%s.heads%2) x.angles.data()[i]=-x.angles.data()[i];
    }
    Tensor dy(x.v.shape.dims);fill(dy,.13f,.031f);
    std::vector<int> valid(s.batch,s.sequence);if (s.batch>1) valid[0]=std::max(0,s.sequence-7);if (s.batch>2) valid.back()=0;
    poison(x,dy,s,valid);
    const auto rx=reference(s,x,valid);
    const auto ri=reference(initial),rs=reference(seed);
    const auto expected=ref::forward(rx,ri);
    const auto dx=ref::backward(rx,ri,values(dy),rs);
    const auto gx=move(x,Device::GPU);
    const auto gi=move(initial,Device::GPU),gs=move(seed,Device::GPU);
    const auto gdy=dy.to(Device::GPU);
    reset_gpu_transfer_stats();
    auto tape=op::Tape::forward(s,gx,gi,valid);
    auto grad=tape->backward(gdy,gs);
    const auto transfer=gpu_transfer_stats();
    require(transfer.d2h_calls==0&&transfer.device_synchronizations==0&&transfer.stream_synchronizations==0,"SISO core introduced a host readback/explicit fence");
    require(tape->audit_status()==std::vector<int>(s.batch,0),"valid SISO operand rejected");
    close(tape->output(),expected.output,"forward");close(tape->snapshot_final_state(),expected.final_state,"final state");
    close(grad.input,dx.input);close(grad.initial_state,dx.initial_state,"initial adjoint");
    const size_t c=1+(s.sequence-1)/s.chunk,bh=static_cast<size_t>(s.batch)*s.heads;
    require(tape->boundary_bytes()==bh*c*(s.head_dim*s.state_dim+s.state_dim+s.head_dim+s.rotary_pairs)*sizeof(float),"retained a dense state history");
    require(tape->replay_bytes()==bh*((s.chunk+1)*s.head_dim*s.state_dim+2*s.chunk*s.state_dim+s.chunk*s.rotary_pairs)*sizeof(float),"replay scratch grew with full sequence");
    rejected([&]{tape->backward(gdy,gs);},"double backward not rejected");
    auto repeat=op::Tape::forward(s,gx,gi,valid);auto again=repeat->backward(gdy,gs);
    require(repeat->audit_status()==std::vector<int>(s.batch,0),"repeat status");
    require(values(tape->output())==values(repeat->output()),"nondeterministic forward");
    for (auto pair:{std::pair{&grad.input.q,&again.input.q},std::pair{&grad.input.k,&again.input.k},std::pair{&grad.input.v,&again.input.v},std::pair{&grad.input.adt,&again.input.adt},
        std::pair{&grad.input.dt,&again.input.dt},std::pair{&grad.input.angles,&again.input.angles},std::pair{&grad.input.q_bias,&again.input.q_bias},std::pair{&grad.initial_state.phase,&again.initial_state.phase},std::pair{&grad.initial_state.ssm,&again.initial_state.ssm}})
        require(values(*pair.first)==values(*pair.second),"nondeterministic VJP");
    std::cout<<"Mamba3 SISO B/S/H/G/P/N/R/chunk="<<s.batch<<'/'<<s.sequence<<'/'<<s.heads<<'/'<<s.groups<<'/'<<s.head_dim<<'/'<<s.state_dim<<'/'<<s.rotary_pairs<<'/'<<s.chunk
        <<" boundary="<<tape->boundary_bytes()<<" replay="<<tape->replay_bytes()<<" partial="<<tape->partial_bytes()<<" phase_wrap="<<phase_wrap<<'\n';
}
Tensor slice(const Tensor& input,int batch,int seq,int start,int length) {
    if (!input.size) return {};
    auto dims=input.shape.dims;const size_t width=input.size/(static_cast<size_t>(batch)*seq);dims[1]=length;
    Tensor result(dims,Device::CPU);const auto cpu=input.to(Device::CPU);
    for (int b=0;b<batch;++b) std::copy_n(cpu.data()+(static_cast<size_t>(b)*seq+start)*width,length*width,result.data()+static_cast<size_t>(b)*length*width);
    return result;
}
op::Operands slice(const op::Operands& x,int batch,int seq,int start,int length) {
    return {slice(x.q,batch,seq,start,length),slice(x.k,batch,seq,start,length),slice(x.v,batch,seq,start,length),slice(x.z,batch,seq,start,length),
        slice(x.adt,batch,seq,start,length),slice(x.dt,batch,seq,start,length),slice(x.trap,batch,seq,start,length),slice(x.angles,batch,seq,start,length),x.q_bias,x.k_bias,x.d};
}
void streaming_and_ownership() {
    op::Shape s{2,35,4,2,7,6,3,16};auto x=operands(s);auto initial=state(s),seed=state(s,true);
    Tensor dy(x.v.shape.dims);fill(dy,.1f,.013f);const std::vector<int> valid{22,35};
    auto gx=move(x,Device::GPU);
    const auto gi=move(initial,Device::GPU),gs=move(seed,Device::GPU);
    const auto gdy=dy.to(Device::GPU);
    auto whole=op::Tape::forward(s,gx,gi,valid);auto dw=whole->backward(gdy,gs);
    auto a=s;a.sequence=16;auto b=s;b.sequence=19;
    auto first=op::Tape::forward(a,move(slice(x,2,35,0,16),Device::GPU),gi,{16,16});
    const auto boundary=first->snapshot_final_state();
    auto last=op::Tape::forward(b,move(slice(x,2,35,16,19),Device::GPU),boundary,{6,19});
    auto db=last->backward(slice(dy,2,35,16,19).to(Device::GPU),gs);
    auto da=first->backward(slice(dy,2,35,0,16).to(Device::GPU),db.initial_state);
    require(whole->audit_status()==std::vector<int>({0,0})&&first->audit_status()==std::vector<int>({0,0})&&last->audit_status()==std::vector<int>({0,0}),"streaming status");
    close(first->output(),values(slice(whole->output(),2,35,0,16)),"stream first");close(last->output(),values(slice(whole->output(),2,35,16,19)),"stream last");
    close(last->snapshot_final_state(),reference(whole->snapshot_final_state()),"stream final state");close(da.initial_state,reference(dw.initial_state),"stream initial adjoint");
    const auto dwa=slice(dw.input,2,35,0,16),dwb=slice(dw.input,2,35,16,19);
    // Shared parameter gradients are sums across calls, not sliced values.
    for (auto pair:{std::pair{&da.input,&dwa},std::pair{&db.input,&dwb}}) {
        close(pair.first->q,values(pair.second->q),"chunk dQ");close(pair.first->k,values(pair.second->k),"chunk dK");close(pair.first->v,values(pair.second->v),"chunk dV");
        close(pair.first->dt,values(pair.second->dt),"chunk dDT");close(pair.first->adt,values(pair.second->adt),"chunk dADT");close(pair.first->angles,values(pair.second->angles),"chunk dAngles");
    }
    for (int kind=0;kind<3;++kind) {
        auto p=values(kind==0?da.input.q_bias:kind==1?da.input.k_bias:da.input.d);
        const auto q=values(kind==0?db.input.q_bias:kind==1?db.input.k_bias:db.input.d);
        for (size_t i=0;i<p.size();++i) p[i]+=q[i];close(kind==0?dw.input.q_bias:kind==1?dw.input.k_bias:dw.input.d,p,"chunk parameter sum");
    }
    // Ownership: caller mutation after forward cannot rewrite its saved inputs.
    auto immutable=op::Tape::forward(s,gx,gi,valid);Tensor poisoned(x.q.shape.dims);fill(poisoned,.01f,.1f,7);
    copy_tensor_bytes(gx.q.raw_data(),Device::GPU,poisoned.data(),Device::CPU,poisoned.size*sizeof(float));
    auto private_grad=immutable->backward(gdy,gs);require(immutable->audit_status()==std::vector<int>({0,0}),"owned tape status");
    close(private_grad.input.q,values(dw.input.q),"owned saved input");
    auto cancelled=op::Tape::forward(s,move(x,Device::GPU),gi,valid);cancelled->cancel();
    rejected([&]{cancelled->backward(gdy);},"cancelled tape reused");
    bool rejected_thread=false;std::thread foreign([&]{try {cancelled->output();} catch (const std::exception&) {rejected_thread=true;}});foreign.join();require(rejected_thread,"foreign tape thread accepted");
    gpu::ExecutionContext other;
    {gpu::ExecutionContext::Scope scope(other);rejected([&]{cancelled->audit_status();},"foreign tape stream accepted");}
    auto bad=s;bad.groups=3;rejected([&]{op::Tape::forward(bad,move(x,Device::GPU));},"invalid geometry accepted");
    auto malformed=move(x,Device::GPU);malformed.q.shape.strides[0]++;
    rejected([&]{op::Tape::forward(s,malformed);},"malformed strides accepted");
    auto partial=gi;partial.phase=Tensor();rejected([&]{op::Tape::forward(s,move(x,Device::GPU),partial);},"partial initial state accepted");
    rejected([&]{op::Tape::forward(s,move(x,Device::GPU),gi,{-1,35});},"invalid prefix accepted");
}
void device_failure_gates() {
    op::Shape s{2,17,2,1,3,4,1,16};auto x=operands(s);auto initial=state(s),seed=state(s,true);
    Tensor dy(x.v.shape.dims);fill(dy,.12f,.11f);
    x.q.data()[0]=std::numeric_limits<float>::quiet_NaN();
    auto failed=op::Tape::forward(s,move(x,Device::GPU),move(initial,Device::GPU));
    auto grad=failed->backward(dy.to(Device::GPU),move(seed,Device::GPU));
    require(failed->audit_status()==std::vector<int>({2,0}),"active NaN did not reject batch");
    const auto out=values(failed->output());for (size_t i=0;i<out.size()/2;++i) require(out[i]==0,"failed output not zero");
    const auto gq=values(grad.input.q);for (size_t i=0;i<gq.size()/2;++i) require(gq[i]==0,"failed gradient not zero");
    for (double v:values(grad.input.q_bias)) require(v==0,"failed batch published shared parameter gradient");
    x=operands(s);auto bad_seed=seed;bad_seed.ssm.data()[0]=std::numeric_limits<float>::infinity();
    auto t=op::Tape::forward(s,move(x,Device::GPU),move(initial,Device::GPU));
    auto dbad=t->backward(dy.to(Device::GPU),move(bad_seed,Device::GPU));
    require(t->audit_status()==std::vector<int>({2,0}),"invalid state seed accepted");
    const auto dstate=values(dbad.initial_state.ssm);for (size_t i=0;i<dstate.size()/2;++i) require(dstate[i]==0,"failed seed leaked adjoint");
    for (int kind=0;kind<4;++kind) {
        auto raw=operands(s);
        if (kind==0) raw.adt.data()[0]=.01f;else if (kind==1) raw.dt.data()[0]=-1;else if (kind==2) raw.dt.data()[0]=17;else raw.v.data()[0]=65;
        auto rejected_op=op::Tape::forward(s,move(raw,Device::GPU));require(rejected_op->audit_status()==std::vector<int>({2,0}),"out of contract operand accepted");
    }
    // Omitted initial state and final seeds must mean zeros, not uninitialized.
    auto zero=op::Tape::forward(s,move(x,Device::GPU));auto dz=zero->backward(dy.to(Device::GPU));
    const auto rx=reference(s,x,{});
    const auto rr=ref::forward(rx);
    const auto dr=ref::backward(rx,{},values(dy));
    require(zero->audit_status()==std::vector<int>({0,0}),"implicit zeros rejected");close(zero->output(),rr.output,"zero initial");close(dz.input,dr.input);close(dz.initial_state,dr.initial_state,"zero state seed");
}
double dot(const std::vector<double>& a,const std::vector<double>& b) {
    double result=0;require(a.size()==b.size(),"dot extent");for (size_t i=0;i<a.size();++i) result+=a[i]*b[i];return result;
}
double objective(const ref::Inputs& x,const ref::State& initial,const std::vector<double>& dy,const ref::State& seed) {
    const auto y=ref::forward(x,initial);
    return dot(y.output,dy)+dot(y.final_state.phase,seed.phase)+dot(y.final_state.ssm,seed.ssm)+dot(y.final_state.k,seed.k)+dot(y.final_state.v,seed.v);
}
void finite_differences() {
    op::Shape s{1,19,4,2,7,6,3,16};auto x=operands(s);auto initial=state(s),seed=state(s,true);
    // Zero DT and very negative ADT are admissible limits; the VJP never
    // divides by DT/alpha, so underflow cannot create an artificial singularity.
    x.dt.data()[0]=0;x.adt.data()[1]=-1000;
    Tensor dy(x.v.shape.dims);fill(dy,.14f,.015f);
    auto rx=reference(s,x,{});
    auto ri=reference(initial);
    const auto rs=reference(seed);
    const auto d=values(dy);
    auto tape=op::Tape::forward(s,move(x,Device::GPU),move(initial,Device::GPU));
    const auto grad=tape->backward(dy.to(Device::GPU),move(seed,Device::GPU));
    require(tape->audit_status()==std::vector<int>({0}),"stable limits rejected");
    auto check=[&](std::vector<double>& input,const Tensor& output,size_t index,const char* name) {
        const double old=input[index],eps=1e-5;input[index]=old+eps;const double plus=objective(rx,ri,d,rs);
        input[index]=old-eps;const double minus=objective(rx,ri,d,rs);input[index]=old;
        const double fd=(plus-minus)/(2*eps),actual=values(output)[index];
        if (std::abs(actual-fd)>3e-4+8e-4*std::abs(fd)) throw std::runtime_error(std::string("GPU finite difference ")+name);
    };
    check(rx.q,grad.input.q,17,"Q");check(rx.k,grad.input.k,11,"K");check(rx.v,grad.input.v,19,"V");
    check(rx.z,grad.input.z,23,"Z");check(rx.dt,grad.input.dt,8,"DT");check(rx.adt,grad.input.adt,9,"ADT");check(rx.trap,grad.input.trap,7,"Trap");
    check(rx.angles,grad.input.angles,21,"Angles");check(rx.q_bias,grad.input.q_bias,8,"QBias");check(rx.k_bias,grad.input.k_bias,7,"KBias");check(rx.d,grad.input.d,2,"D");
    check(ri.phase,grad.initial_state.phase,5,"initial phase");check(ri.ssm,grad.initial_state.ssm,19,"initial SSM");check(ri.k,grad.initial_state.k,7,"initial K");check(ri.v,grad.initial_state.v,9,"initial V");
}
void raw_bounds_and_aliases() {
#ifdef USE_CUDA
    op::Shape s{2,33,4,2,7,6,3,16};const auto input=move(operands(s),Device::GPU);
    cuda_detail::DeviceBuffer<int> valid,status;require(valid.ensure(s.batch)&&status.ensure(s.batch),"metadata allocation");
    const int prefixes[2]={33,0};require(cudaMemcpy(valid.get(),prefixes,sizeof(prefixes),cudaMemcpyHostToDevice)==cudaSuccess,"raw prefix upload");
    mg::Input x{input.q.raw_data(),input.k.raw_data(),input.v.raw_data(),input.z.raw_data(),input.adt.raw_data(),input.dt.raw_data(),input.trap.raw_data(),
        input.angles.raw_data(),input.q_bias.raw_data(),input.k_bias.raw_data(),input.d.raw_data(),valid.get()};
    auto guarded=[](size_t n) {Tensor h({static_cast<int>(n+4)});std::fill_n(h.data(),h.size,123456.0f);return h.to(Device::GPU);};
    Tensor out=guarded(input.v.size),boundary=guarded(mg::boundary_elements(s)),replay=guarded(mg::replay_elements(s)),partial=guarded(mg::gradient_partial_elements(s));
    const size_t bh=static_cast<size_t>(s.batch)*s.heads;
    op::State final{guarded(bh*s.rotary_pairs),guarded(bh*s.head_dim*s.state_dim),guarded(bh*s.state_dim),guarded(bh*s.head_dim)};
    auto dest=[](op::State& st) {return mg::State{st.phase.raw_data(),st.ssm.raw_data(),st.k.raw_data(),st.v.raw_data()};};
    require(!mg::forward(s,x,{},dest(final),const_cast<float*>(x.v),boundary.raw_data(),status.get()),"raw aliased forward accepted");
    require(mg::forward(s,x,{},dest(final),out.raw_data(),boundary.raw_data(),status.get()),"raw forward enqueue");
    op::Operands dx{guarded(input.q.size),guarded(input.k.size),guarded(input.v.size),guarded(input.z.size),guarded(input.adt.size),guarded(input.dt.size),guarded(input.trap.size),
        guarded(input.angles.size),guarded(input.q_bias.size),guarded(input.k_bias.size),guarded(input.d.size)};
    op::State di{guarded(bh*s.rotary_pairs),guarded(bh*s.head_dim*s.state_dim),guarded(bh*s.state_dim),guarded(bh*s.head_dim)};
    mg::Gradient grad{dx.q.raw_data(),dx.k.raw_data(),dx.v.raw_data(),dx.z.raw_data(),dx.adt.raw_data(),dx.dt.raw_data(),dx.trap.raw_data(),dx.angles.raw_data(),dx.q_bias.raw_data(),dx.k_bias.raw_data(),dx.d.raw_data()};
    Tensor dy(input.v.shape.dims);fill(dy,.1f,.017f);const auto gdy=dy.to(Device::GPU);
    auto alias=grad;alias.k=alias.q;
    require(!mg::backward(s,x,{},gdy.raw_data(),{},alias,dest(di),boundary.raw_data(),replay.raw_data(),partial.raw_data(),status.get()),"raw aliased gradient accepted");
    require(mg::backward(s,x,{},gdy.raw_data(),{},grad,dest(di),boundary.raw_data(),replay.raw_data(),partial.raw_data(),status.get()),"raw backward enqueue");
    nsos::gpu_parity_test::cuda_sync_or_throw("raw guarded Mamba3");
    int issues[2];require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"raw status download");require(issues[0]==0&&issues[1]==0,"raw status");
    for (auto* t:{&out,&boundary,&replay,&partial,&final.phase,&final.ssm,&final.k,&final.v,&di.phase,&di.ssm,&di.k,&di.v,
        &dx.q,&dx.k,&dx.v,&dx.z,&dx.adt,&dx.dt,&dx.trap,&dx.angles,&dx.q_bias,&dx.k_bias,&dx.d}) {
        const auto v=values(*t);for (size_t i=v.size()-4;i<v.size();++i) require(v[i]==123456.0,"raw write exceeded declared capacity");
    }
    const float nan=std::numeric_limits<float>::quiet_NaN();
    require(cudaMemcpy(boundary.raw_data()+mg::chunks(s)*s.rotary_pairs,&nan,sizeof(nan),cudaMemcpyHostToDevice)==cudaSuccess,"boundary poison upload");
    require(mg::backward(s,x,{},gdy.raw_data(),{},grad,dest(di),boundary.raw_data(),replay.raw_data(),partial.raw_data(),status.get()),"poisoned replay enqueue");
    nsos::gpu_parity_test::cuda_sync_or_throw("raw numeric replay gate");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"numeric replay status");require(issues[0]==3&&issues[1]==0,"replay numeric failure not latched");
    const auto failed_dq=values(dx.q);for (size_t i=0;i<input.q.size/2;++i) require(failed_dq[i]==0,"numeric replay published invalid gradient");
    // Raw device-prefix validation must fail closed without dereferencing an
    // invalid length or exposing a half-written entering boundary.
    const int bad[2]={-1,34};require(cudaMemcpy(valid.get(),bad,sizeof(bad),cudaMemcpyHostToDevice)==cudaSuccess,"bad prefix upload");
    require(mg::forward(s,x,{},dest(final),out.raw_data(),boundary.raw_data(),status.get()),"bad-prefix enqueue rejected prematurely");
    nsos::gpu_parity_test::cuda_sync_or_throw("raw invalid prefixes");
    require(cudaMemcpy(issues,status.get(),sizeof(issues),cudaMemcpyDeviceToHost)==cudaSuccess,"bad prefix status");require(issues[0]==1&&issues[1]==1,"device prefix gate missed invalid value");
    const auto zeros=values(out);for (size_t i=0;i<input.v.size;++i) require(zeros[i]==0,"invalid device prefix leaked output");
#endif
}
}
int main() {
    return nsos::gpu_parity_test::run_parity("mamba3_siso",[] {
        parity({3,1,4,2,3,4,1,16});parity({2,17,4,1,7,6,3,16});
        parity({2,33,4,2,16,32,8,32});parity({2,65,2,2,64,64,16,16});
        parity({2,65,2,1,64,64,32,32});parity({1,129,4,1,16,32,16,32});
        parity({1,257,2,1,32,64,32,32});parity({2,35,4,2,7,6,1,16},false,false);
        parity({2,65,4,2,7,6,3,16},true,true,true);
        {
            gpu::ExecutionContext lane;
            gpu::ExecutionContext::Scope scope(lane);
            parity({2,33,4,2,7,6,3,32},true,true,true);
        }
        streaming_and_ownership();device_failure_gates();finite_differences();raw_bounds_and_aliases();
    });
}
