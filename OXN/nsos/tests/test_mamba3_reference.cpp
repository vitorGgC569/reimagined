#include "mamba3_reference.h"
#include "mamba3_preprocessing.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace m3=nsos::mamba3_reference;
namespace {
void require(bool ok,const std::string& name) {if (!ok) throw std::runtime_error(name);}
void close(double a,double b,const std::string& name,double atol=2e-9,double rtol=2e-7) {
    if (!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>atol+rtol*std::max(std::abs(a),std::abs(b)))
        throw std::runtime_error(name+": "+std::to_string(a)+" versus "+std::to_string(b));
}
void close(const std::vector<double>& a,const std::vector<double>& b,const std::string& name) {
    require(a.size()==b.size(),name+" shape");
    for (size_t i=0;i<a.size();++i) close(a[i],b[i],name+"["+std::to_string(i)+"]");
}
void close(const m3::State& a,const m3::State& b,const std::string& name) {
    close(a.phase,b.phase,name+" phase");close(a.ssm,b.ssm,name+" ssm");
    close(a.k,b.k,name+" K");close(a.v,b.v,name+" V");
}
double dot(const std::vector<double>& a,const std::vector<double>& b) {
    require(a.size()==b.size(),"dot shape");double out=0;
    for (size_t i=0;i<a.size();++i) out+=a[i]*b[i];return out;
}
double dot(const m3::State& a,const m3::State& b) {
    return dot(a.phase,b.phase)+dot(a.ssm,b.ssm)+dot(a.k,b.k)+dot(a.v,b.v);
}
void fill(std::vector<double>& v,size_t count,double scale,int salt) {
    v.resize(count);for (size_t i=0;i<count;++i) v[i]=scale*(static_cast<int>((i*7+salt)%19)-9);
}
m3::Inputs fixture(int rotary=1,bool gate=true,bool skip=true) {
    m3::Inputs x;x.geometry={2,3,2,1,3,4,rotary};
    fill(x.q,24,.043,1);fill(x.k,24,.051,5);fill(x.v,36,.032,7);
    if (gate) fill(x.z,36,.17,2);
    fill(x.adt,12,.004,1);for (auto& a:x.adt) a-=.19;
    fill(x.dt,12,.005,8);for (auto& d:x.dt) d+=.13;
    fill(x.trap,12,.22,11);fill(x.angles,12*rotary,.033,3);
    fill(x.q_bias,8,.08,4);fill(x.k_bias,8,.07,6);
    if (skip) fill(x.d,2,.11,17);
    x.valid_lengths={2,3};return x;
}
m3::State initial(const m3::Geometry& g) {
    auto s=m3::zero_state(g);
    fill(s.phase,s.phase.size(),.04,7); // includes negative phases, away from modulo boundary
    fill(s.ssm,s.ssm.size(),.012,3);fill(s.k,s.k.size(),.027,2);fill(s.v,s.v.size(),.035,8);
    return s;
}
m3::State seed(const m3::Geometry& g) {
    auto s=m3::zero_state(g);
    fill(s.phase,s.phase.size(),.013,9);fill(s.ssm,s.ssm.size(),.017,4);
    fill(s.k,s.k.size(),.023,5);fill(s.v,s.v.size(),.026,12);return s;
}
// Independent quadratic primal. Constructs the dense semiseparable mixing
// matrix directly instead of updating the SSM recurrence used by the port.
m3::Forward quadratic(const m3::Inputs& x,const m3::State& s0) {
    const auto g=x.geometry;
    const double pi=std::numbers::pi_v<double>;
    m3::Forward y;y.output.resize(x.v.size());y.final_state=s0;
    for (int b=0;b<g.batch;++b) for (int h=0;h<g.heads;++h) {
        const size_t bh=static_cast<size_t>(b)*g.heads+h;
        const int length=x.valid_lengths.empty()?g.sequence:x.valid_lengths[b];
        std::vector<double> alpha(length),beta(length),gamma(length),q(length*g.state_dim),k(q.size());
        auto phase=std::vector<double>(s0.phase.begin()+bh*g.rotary_pairs,s0.phase.begin()+(bh+1)*g.rotary_pairs);
        for (int t=0;t<length;++t) {
            const size_t th=(static_cast<size_t>(b)*g.sequence+t)*g.heads+h;
            const double lambda=1/(1+std::exp(-x.trap[th]));
            alpha[t]=std::exp(x.adt[th]);beta[t]=(1-lambda)*x.dt[th]*alpha[t];gamma[t]=lambda*x.dt[th];
            for (int r=0;r<g.rotary_pairs;++r) phase[r]+=pi*std::tanh(x.angles[th*g.rotary_pairs+r])*x.dt[th];
            for (int n=0;n<g.state_dim;n+=2) {
                const size_t group=((static_cast<size_t>(b)*g.sequence+t)*g.groups+h/(g.heads/g.groups))*g.state_dim+n;
                const size_t hi=static_cast<size_t>(h)*g.state_dim+n;
                const double c=std::cos(n/2<g.rotary_pairs?phase[n/2]:0),s=std::sin(n/2<g.rotary_pairs?phase[n/2]:0);
                for (int kind=0;kind<2;++kind) {
                    const auto& raw=kind?x.k:x.q;const auto& bias=kind?x.k_bias:x.q_bias;auto& rot=kind?k:q;
                    const double a=raw[group]+bias[hi],d=raw[group+1]+bias[hi+1];
                    rot[t*g.state_dim+n]=c*a-s*d;rot[t*g.state_dim+n+1]=s*a+c*d;
                }
            }
        }
        for (int r=0;r<g.rotary_pairs;++r) {
            if (length) y.final_state.phase[bh*g.rotary_pairs+r]=phase[r]-2*pi*std::floor(phase[r]/(2*pi));
        }
        for (int t=0;t<length;++t) for (int p=0;p<g.head_dim;++p) {
            const size_t vp=((static_cast<size_t>(b)*g.sequence+t)*g.heads+h)*g.head_dim+p;
            double value=0;
            for (int n=0;n<g.state_dim;++n) {
                const size_t si=(bh*g.head_dim+p)*g.state_dim+n;
                double initial_decay=1;for (int j=0;j<=t;++j) initial_decay*=alpha[j];
                double hstate=initial_decay*s0.ssm[si];
                double previous_decay=beta[0];for (int j=1;j<=t;++j) previous_decay*=alpha[j];
                hstate+=previous_decay*s0.k[bh*g.state_dim+n]*s0.v[bh*g.head_dim+p];
                for (int j=0;j<=t;++j) {
                    double coefficient;
                    if (j==t) coefficient=gamma[j];
                    else {coefficient=gamma[j]*alpha[j+1]+beta[j+1];for (int l=j+2;l<=t;++l) coefficient*=alpha[l];}
                    const size_t jvp=((static_cast<size_t>(b)*g.sequence+j)*g.heads+h)*g.head_dim+p;
                    hstate+=coefficient*k[j*g.state_dim+n]*x.v[jvp];
                }
                value+=hstate*q[t*g.state_dim+n];
                if (t==length-1) y.final_state.ssm[si]=hstate;
            }
            if (!x.d.empty()) value+=x.d[h]*x.v[vp];
            if (!x.z.empty()) value*=x.z[vp]/(1+std::exp(-x.z[vp]));
            y.output[vp]=value;
        }
        if (length) {
            std::copy_n(k.data()+(length-1)*g.state_dim,g.state_dim,y.final_state.k.data()+bh*g.state_dim);
            const size_t vp=((static_cast<size_t>(b)*g.sequence+length-1)*g.heads+h)*g.head_dim;
            std::copy_n(x.v.data()+vp,g.head_dim,y.final_state.v.data()+bh*g.head_dim);
        }
    }
    return y;
}
double objective(const m3::Inputs& x,const m3::State& s,const std::vector<double>& dy,const m3::State& ds) {
    const auto y=quadratic(x,s);return dot(y.output,dy)+dot(y.final_state,ds);
}
void finite_difference(int rotary,bool gate,bool skip) {
    auto x=fixture(rotary,gate,skip);auto s=initial(x.geometry),ds=seed(x.geometry);
    std::vector<double> dy;fill(dy,x.v.size(),.019,10);
    const auto f=m3::forward(x,s),ref=quadratic(x,s);
    close(f.output,ref.output,"quadratic output");close(f.final_state,ref.final_state,"quadratic state");
    const auto dx=m3::backward(x,s,dy,ds);
    const double eps=1e-6;
    auto check=[&](std::vector<double>& values,const std::vector<double>& analytic,const std::string& name) {
        for (size_t i=0;i<values.size();++i) {
            const double saved=values[i];values[i]=saved+eps;const double plus=objective(x,s,dy,ds);
            values[i]=saved-eps;const double minus=objective(x,s,dy,ds);values[i]=saved;
            close(analytic[i],(plus-minus)/(2*eps),name+"["+std::to_string(i)+"]",2e-8,3e-6);
        }
    };
    check(x.q,dx.input.q,"dQ");check(x.k,dx.input.k,"dK");check(x.v,dx.input.v,"dV");check(x.z,dx.input.z,"dZ");
    check(x.dt,dx.input.dt,"dDT");check(x.adt,dx.input.adt,"dADT");check(x.trap,dx.input.trap,"dTrap");
    check(x.angles,dx.input.angles,"dAngles");check(x.q_bias,dx.input.q_bias,"dQBias");check(x.k_bias,dx.input.k_bias,"dKBias");check(x.d,dx.input.d,"dD");
    check(s.phase,dx.initial_state.phase,"dInitialPhase");check(s.ssm,dx.initial_state.ssm,"dInitialSSM");
    check(s.k,dx.initial_state.k,"dInitialK");check(s.v,dx.initial_state.v,"dInitialV");
}
std::vector<double> slice(const std::vector<double>& v,int batch,int seq,int width,int start,int count) {
    if (v.empty()) return {};
    std::vector<double> out(static_cast<size_t>(batch)*count*width);
    for (int b=0;b<batch;++b) std::copy_n(v.data()+(static_cast<size_t>(b)*seq+start)*width,count*width,out.data()+static_cast<size_t>(b)*count*width);
    return out;
}
m3::Inputs slice(const m3::Inputs& x,int start,int count) {
    auto out=x;const auto g=x.geometry;out.geometry.sequence=count;
    out.q=slice(x.q,g.batch,g.sequence,g.groups*g.state_dim,start,count);out.k=slice(x.k,g.batch,g.sequence,g.groups*g.state_dim,start,count);
    out.v=slice(x.v,g.batch,g.sequence,g.heads*g.head_dim,start,count);out.z=slice(x.z,g.batch,g.sequence,g.heads*g.head_dim,start,count);
    out.dt=slice(x.dt,g.batch,g.sequence,g.heads,start,count);out.adt=slice(x.adt,g.batch,g.sequence,g.heads,start,count);out.trap=slice(x.trap,g.batch,g.sequence,g.heads,start,count);
    out.angles=slice(x.angles,g.batch,g.sequence,g.heads*g.rotary_pairs,start,count);
    if (!out.valid_lengths.empty()) for (auto& n:out.valid_lengths) n=std::clamp(n-start,0,count);
    return out;
}
void streaming_bptt() {
    auto x=fixture(2);auto s=initial(x.geometry),ds=seed(x.geometry);
    std::vector<double> dy;fill(dy,x.v.size(),.019,10);
    const auto whole=m3::forward(x,s);
    const auto a=slice(x,0,1),b=slice(x,1,2);
    const auto first=m3::forward(a,s),last=m3::forward(b,first.final_state);
    close(first.output,slice(whole.output,2,3,6,0,1),"stream first");close(last.output,slice(whole.output,2,3,6,1,2),"stream last");close(last.final_state,whole.final_state,"stream state");
    const auto dw=m3::backward(x,s,dy,ds),db=m3::backward(b,first.final_state,slice(dy,2,3,6,1,2),ds);
    const auto da=m3::backward(a,s,slice(dy,2,3,6,0,1),db.initial_state);
    close(da.initial_state,dw.initial_state,"cross chunk state adjoint");
    const auto dfirst=slice(dw.input,0,1),dlast=slice(dw.input,1,2);
    for (auto triple:{std::pair{&da.input,&dfirst},std::pair{&db.input,&dlast}}) {
        close(triple.first->q,triple.second->q,"chunk dQ");close(triple.first->k,triple.second->k,"chunk dK");close(triple.first->v,triple.second->v,"chunk dV");
        close(triple.first->z,triple.second->z,"chunk dZ");close(triple.first->dt,triple.second->dt,"chunk dDT");close(triple.first->adt,triple.second->adt,"chunk dADT");
        close(triple.first->angles,triple.second->angles,"chunk dAngles");close(triple.first->trap,triple.second->trap,"chunk dTrap");
    }
    auto sum=[](std::vector<double> a,const std::vector<double>& b) {for (size_t i=0;i<a.size();++i) a[i]+=b[i];return a;};
    close(sum(da.input.q_bias,db.input.q_bias),dw.input.q_bias,"chunk dQBias");close(sum(da.input.k_bias,db.input.k_bias),dw.input.k_bias,"chunk dKBias");close(sum(da.input.d,db.input.d),dw.input.d,"chunk dD");
}
template<class F> void rejected(F fn,const std::string& name) {
    bool threw=false;try {fn();} catch (const std::invalid_argument&) {threw=true;} catch (const std::length_error&) {threw=true;}
    require(threw,name+" not rejected");
}
void masks_and_contracts() {
    auto x=fixture();x.valid_lengths={0,2};auto s=initial(x.geometry);auto ds=seed(x.geometry);
    const auto clean=m3::forward(x,s);std::vector<double> dy(x.v.size(),.23);
    const double poison=std::numeric_limits<double>::quiet_NaN();
    for (int b=0;b<2;++b) for (int t=x.valid_lengths[b];t<3;++t) {
        for (auto pair:{std::pair{&x.q,4},std::pair{&x.k,4},std::pair{&x.v,6},std::pair{&x.z,6},std::pair{&x.dt,2},std::pair{&x.adt,2},std::pair{&x.trap,2},std::pair{&x.angles,2},std::pair{&dy,6}})
            std::fill_n(pair.first->data()+(b*3+t)*pair.second,pair.second,poison);
    }
    const auto poisoned=m3::forward(x,s);
    const auto dx=m3::backward(x,s,dy,ds);
    close(poisoned.output,clean.output,"poisoned padding output");close(poisoned.final_state,clean.final_state,"poisoned padding state");
    for (auto pair:{std::pair{&dx.input.q,4},std::pair{&dx.input.k,4},std::pair{&dx.input.v,6},std::pair{&dx.input.z,6},std::pair{&dx.input.dt,2},std::pair{&dx.input.adt,2},std::pair{&dx.input.trap,2},std::pair{&dx.input.angles,2}}) {
        for (double v:*pair.first) require(std::isfinite(v),"poison leaked into VJP");
        for (int b=0;b<2;++b) for (int t=x.valid_lengths[b];t<3;++t) for (int n=0;n<pair.second;++n) require((*pair.first)[(b*3+t)*pair.second+n]==0,"padding nonzero gradient");
    }
    for (auto pair:{std::pair{&dx.initial_state.phase,&ds.phase},std::pair{&dx.initial_state.ssm,&ds.ssm},std::pair{&dx.initial_state.k,&ds.k},std::pair{&dx.initial_state.v,&ds.v}})
        for (size_t i=0;i<pair.first->size()/2;++i) close((*pair.first)[i],(*pair.second)[i],"zero length state carry");
    auto good=fixture();auto bad=good;bad.geometry.groups=3;rejected([&]{m3::forward(bad);},"nondivisible groups");
    bad=good;bad.geometry.rotary_pairs=3;rejected([&]{m3::forward(bad);},"rotary geometry");
    bad=good;bad.geometry.batch=0;rejected([&]{m3::forward(bad);},"zero geometry");
    bad=good;bad.q.pop_back();rejected([&]{m3::forward(bad);},"Q shape");
    bad=good;bad.valid_lengths={-1,3};rejected([&]{m3::forward(bad);},"negative prefix");
    bad=good;bad.valid_lengths={1,4};rejected([&]{m3::forward(bad);},"oversized prefix");
    auto partial=initial(good.geometry);partial.phase.clear();rejected([&]{m3::forward(good,partial);},"partial initial state");
    rejected([&]{m3::backward(good,{},{});},"output seed shape");
    partial=seed(good.geometry);partial.v.clear();rejected([&]{m3::backward(good,{},std::vector<double>(good.v.size()),partial);},"partial final seed");
    const auto implicit=m3::forward(good),explicit_zero=m3::forward(good,m3::zero_state(good.geometry));
    close(implicit.output,explicit_zero.output,"implicit zero state");close(implicit.final_state,explicit_zero.final_state,"implicit zero final");
}
void preprocessing_vjp() {
    namespace pre=nsos::mamba3_preprocessing;
    const auto base=fixture(2);pre::Inputs x;
    x.geometry=base.geometry;x.valid_lengths=base.valid_lengths;
    x.q=base.q;x.k=base.k;x.v=base.v;x.z=base.z;x.trap=base.trap;
    x.q_bias=base.q_bias;x.k_bias=base.k_bias;x.d=base.d;
    fill(x.raw_a,12,.23,1);fill(x.raw_dt,12,.13,6);fill(x.angles,12,.09,8);
    fill(x.q_norm,4,.02,2);for (auto& v:x.q_norm) v+=1;
    fill(x.k_norm,4,.02,5);for (auto& v:x.k_norm) v+=1;
    fill(x.dt_bias,2,.07,3);x.a_floor=.3;
    x.raw_a[0]=-9; // active heavy-tail floor, zero derivative
    auto s=initial(x.geometry),ds=seed(x.geometry);std::vector<double> dy;fill(dy,x.v.size(),.017,7);
    const auto processed=pre::forward(x);
    // Independent original preprocessing equations, including head broadcast,
    // normalize-before-bias and stable softplus at extreme inputs.
    for (int b=0;b<2;++b) for (int t=0;t<x.valid_lengths[b];++t) {
        const int row=b*3+t;
        for (int h=0;h<2;++h) {
            const int th=row*2+h;const double u=x.raw_a[th],a=-std::max(.3,u>=0?1+u:1/(1-u));
            const double dt=std::log(1+std::exp(x.raw_dt[th]+x.dt_bias[h]));
            close(processed.dt[th],dt,"softplus");close(processed.adt[th],a*dt,"heavy-tail ADT");
            for (int r=0;r<2;++r) close(processed.angles[th*2+r],x.angles[row*2+r],"head angle broadcast");
        }
        for (auto pair:{std::pair{&x.q,&x.q_norm},std::pair{&x.k,&x.k_norm}}) {
            double square=0;for (int n=0;n<4;++n) square+=(*pair.first)[row*4+n]*(*pair.first)[row*4+n];
            const auto& result=pair.first==&x.q?processed.q:processed.k;
            for (int n=0;n<4;++n) close(result[row*4+n],(*pair.first)[row*4+n]*(*pair.second)[n]/std::sqrt(square/4+1e-5),"BCNorm");
        }
    }
    const auto dq=m3::backward(processed,s,dy,ds);const auto dx=pre::backward(x,dq.input);
    const double eps=1e-6;
    auto check=[&](std::vector<double>& values,const std::vector<double>& analytic,const char* name) {
        for (size_t i=0;i<values.size();++i) {
            const double saved=values[i];values[i]=saved+eps;const double plus=objective(pre::forward(x),s,dy,ds);
            values[i]=saved-eps;const double minus=objective(pre::forward(x),s,dy,ds);values[i]=saved;
            close(analytic[i],(plus-minus)/(2*eps),std::string("preprocess ")+name+"["+std::to_string(i)+"]",8e-8,4e-6);
        }
    };
    check(x.q,dx.q,"Q");check(x.k,dx.k,"K");check(x.v,dx.v,"V");check(x.z,dx.z,"Z");
    check(x.raw_a,dx.raw_a,"raw A");check(x.raw_dt,dx.raw_dt,"raw DT");check(x.trap,dx.trap,"Trap");check(x.angles,dx.angles,"Angles");
    check(x.q_norm,dx.q_norm,"Q Norm");check(x.k_norm,dx.k_norm,"K Norm");check(x.dt_bias,dx.dt_bias,"DT Bias");
    check(x.q_bias,dx.q_bias,"Q Bias");check(x.k_bias,dx.k_bias,"K Bias");check(x.d,dx.d,"D");
    auto bad=x;bad.angles.pop_back();rejected([&]{pre::forward(bad);},"shared angles shape");
    bad=x;bad.norm_eps=0;rejected([&]{pre::forward(bad);},"normalization epsilon");
    auto adj=dq.input;adj.geometry.groups=2;rejected([&]{pre::backward(x,adj);},"adjoint metadata");
    // Padding must remain unread even before BCNorm/softplus/angle broadcast.
    x.valid_lengths={0,2};
    for (int b=0;b<2;++b) for (int t=x.valid_lengths[b];t<3;++t) {
        for (auto pair:{std::pair{&x.q,4},std::pair{&x.k,4},std::pair{&x.v,6},std::pair{&x.z,6},std::pair{&x.raw_a,2},std::pair{&x.raw_dt,2},std::pair{&x.trap,2},std::pair{&x.angles,2}})
            std::fill_n(pair.first->data()+(b*3+t)*pair.second,pair.second,std::numeric_limits<double>::quiet_NaN());
    }
    const auto masked=pre::forward(x);
    const auto masked_grad=pre::backward(x,m3::backward(masked,s,dy,ds).input);
    for (auto* v:{&masked.q,&masked.k,&masked.v,&masked.z,&masked.adt,&masked.dt,&masked.trap,&masked.angles,&masked_grad.q,&masked_grad.k,&masked_grad.raw_a,&masked_grad.raw_dt,&masked_grad.angles})
        for (double value:*v) require(std::isfinite(value),"preprocessing consumed poison padding");
    x.raw_dt[6]=1000;x.raw_dt[7]=-1000;const auto extreme=pre::forward(x);
    require(std::isfinite(extreme.dt[6])&&extreme.dt[6]>999&&extreme.dt[7]==0,"stable softplus extremes");
}
}
int main() {
    try {finite_difference(1,true,true);finite_difference(2,true,true);finite_difference(1,false,false);streaming_bptt();masks_and_contracts();preprocessing_vjp();
        std::cout<<"Mamba3 SISO FP64 quadratic/VJP/streaming/mask contracts passed\n";return 0;
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
