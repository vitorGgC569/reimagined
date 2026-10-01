#include "mamba3_reference.h"
#include "mamba3_layer_math.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace nsos::mamba3_reference {
namespace {
constexpr double pi=std::numbers::pi_v<double>,two_pi=2*pi;
size_t extent(std::initializer_list<int> dims) {
    size_t n=1;
    for (int d:dims) {
        if (d<=0 || n>std::numeric_limits<size_t>::max()/static_cast<size_t>(d))
            throw std::invalid_argument("Mamba3 reference extent overflow/nonpositive dimension");
        n*=static_cast<size_t>(d);
    }
    if (n>std::vector<double>().max_size()) throw std::length_error("Mamba3 reference vector too large");
    return n;
}
void geometry(const Geometry& g) {
    extent({g.batch,g.sequence,g.heads,g.groups,g.head_dim,g.state_dim,g.rotary_pairs});
    if (g.heads%g.groups || g.state_dim%2 || g.rotary_pairs>g.state_dim/2 ||
        g.sequence==std::numeric_limits<int>::max())
        throw std::invalid_argument("Mamba3 reference invalid groups/rotary/history geometry");
}
void shape(const std::vector<double>& x,size_t n,const char* name) {
    if (x.size()!=n) throw std::invalid_argument(std::string("Mamba3 reference shape: ")+name);
}
bool empty(const State& s) {return s.phase.empty()&&s.ssm.empty()&&s.k.empty()&&s.v.empty();}
void state_shape(const State& s,const Geometry& g) {
    shape(s.phase,extent({g.batch,g.heads,g.rotary_pairs}),"phase state");
    shape(s.ssm,extent({g.batch,g.heads,g.head_dim,g.state_dim}),"SSM state");
    shape(s.k,extent({g.batch,g.heads,g.state_dim}),"previous K state");
    shape(s.v,extent({g.batch,g.heads,g.head_dim}),"previous V state");
}
void validate(const Inputs& x) {
    const auto& g=x.geometry; geometry(g);
    shape(x.q,extent({g.batch,g.sequence,g.groups,g.state_dim}),"Q"); shape(x.k,x.q.size(),"K");
    shape(x.v,extent({g.batch,g.sequence,g.heads,g.head_dim}),"V");
    if (!x.z.empty()) shape(x.z,x.v.size(),"Z");
    shape(x.adt,extent({g.batch,g.sequence,g.heads}),"ADT"); shape(x.dt,x.adt.size(),"DT"); shape(x.trap,x.adt.size(),"Trap");
    shape(x.angles,extent({g.batch,g.sequence,g.heads,g.rotary_pairs}),"angles");
    shape(x.q_bias,extent({g.heads,g.state_dim}),"Q bias"); shape(x.k_bias,x.q_bias.size(),"K bias");
    if (!x.d.empty()) shape(x.d,static_cast<size_t>(g.heads),"D");
    if (!x.valid_lengths.empty()) {
        if (x.valid_lengths.size()!=static_cast<size_t>(g.batch)) throw std::invalid_argument("Mamba3 valid-prefix shape");
        for (int length:x.valid_lengths) if (length<0 || length>g.sequence) throw std::invalid_argument("Mamba3 valid prefix out of range");
    }
}
double sigmoid(double x) {if (x>=0) return 1/(1+std::exp(-x));const double e=std::exp(x);return e/(1+e);}
size_t time_head(const Geometry& g,int b,int t,int h) {return (static_cast<size_t>(b)*g.sequence+t)*g.heads+h;}
size_t qindex(const Geometry& g,int b,int t,int h,int n) {return ((static_cast<size_t>(b)*g.sequence+t)*g.groups+h/(g.heads/g.groups))*g.state_dim+n;}
size_t state_index(const Geometry& g,int b,int h,int p,int n) {return ((static_cast<size_t>(b)*g.heads+h)*g.head_dim+p)*g.state_dim+n;}
size_t history_index(const Geometry& g,int b,int h,int t,int p,int n) {return (((static_cast<size_t>(b)*g.heads+h)*(g.sequence+1)+t)*g.head_dim+p)*g.state_dim+n;}
struct Trace {Forward result;std::vector<double> history,qr,kr,phase;};
Trace evaluate(const Inputs& x,const State& initial,bool save_history) {
    validate(x);const auto& g=x.geometry;
    Trace tr;
    tr.result.final_state=empty(initial)?zero_state(g):initial;
    state_shape(tr.result.final_state,g);
    tr.result.output.assign(x.v.size(),0);
    if (save_history) tr.history.resize(extent({g.batch,g.heads,g.sequence+1,g.head_dim,g.state_dim}));
    tr.qr.resize(extent({g.batch,g.sequence,g.heads,g.state_dim})); tr.kr.resize(tr.qr.size());
    tr.phase.resize(x.angles.size());
    auto& state=tr.result.final_state;
    for (int b=0;b<g.batch;++b) for (int h=0;h<g.heads;++h) {
        const size_t bh=static_cast<size_t>(b)*g.heads+h;
        if (save_history) for (int p=0;p<g.head_dim;++p) for (int n=0;n<g.state_dim;++n)
            tr.history[history_index(g,b,h,0,p,n)]=state.ssm[state_index(g,b,h,p,n)];
        const int length=x.valid_lengths.empty()?g.sequence:x.valid_lengths[b];
        for (int t=0;t<length;++t) {
            const size_t th=time_head(g,b,t,h),kn=th*g.state_dim,vp=th*g.head_dim;
            const double alpha=std::exp(x.adt[th]),lambda=sigmoid(x.trap[th]);
            const double beta=(1-lambda)*x.dt[th]*alpha,gamma=lambda*x.dt[th];
            for (int r=0;r<g.rotary_pairs;++r) {
                double& phase=state.phase[bh*g.rotary_pairs+r];
                phase+=pi*std::tanh(x.angles[th*g.rotary_pairs+r])*x.dt[th];
                phase-=two_pi*std::floor(phase/two_pi);
                tr.phase[th*g.rotary_pairs+r]=phase;
            }
            for (int n=0;n<g.state_dim;n+=2) {
                const int r=n/2;const double phase=r<g.rotary_pairs?state.phase[bh*g.rotary_pairs+r]:0;
                const double c=std::cos(phase),s=std::sin(phase);
                for (int kind=0;kind<2;++kind) {
                    const auto& values=kind?x.k:x.q;const auto& bias=kind?x.k_bias:x.q_bias;
                    auto& out=kind?tr.kr:tr.qr;
                    const double a=values[qindex(g,b,t,h,n)]+bias[static_cast<size_t>(h)*g.state_dim+n];
                    const double d=values[qindex(g,b,t,h,n+1)]+bias[static_cast<size_t>(h)*g.state_dim+n+1];
                    out[kn+n]=a*c-d*s;out[kn+n+1]=a*s+d*c;
                }
            }
            for (int p=0;p<g.head_dim;++p) {
                double y=0;
                for (int n=0;n<g.state_dim;++n) {
                    double& hs=state.ssm[state_index(g,b,h,p,n)];
                    hs=alpha*hs+beta*state.k[bh*g.state_dim+n]*state.v[bh*g.head_dim+p]+
                        gamma*tr.kr[kn+n]*x.v[vp+p];
                    if (save_history) tr.history[history_index(g,b,h,t+1,p,n)]=hs;
                    y+=hs*tr.qr[kn+n];
                }
                if (!x.d.empty()) y+=x.d[h]*x.v[vp+p];
                if (!x.z.empty()) y*=x.z[vp+p]*sigmoid(x.z[vp+p]);
                tr.result.output[vp+p]=y;
            }
            std::copy_n(tr.kr.data()+kn,g.state_dim,state.k.data()+bh*g.state_dim);
            std::copy_n(x.v.data()+vp,g.head_dim,state.v.data()+bh*g.head_dim);
        }
    }
    return tr;
}
Inputs zeros_like(const Inputs& x) {
    Inputs out;out.geometry=x.geometry;out.valid_lengths=x.valid_lengths;
    for (auto pair:{std::pair{&out.q,&x.q},{&out.k,&x.k},{&out.v,&x.v},{&out.z,&x.z},
        {&out.dt,&x.dt},{&out.adt,&x.adt},{&out.trap,&x.trap},{&out.angles,&x.angles},
        {&out.q_bias,&x.q_bias},{&out.k_bias,&x.k_bias},{&out.d,&x.d}}) pair.first->assign(pair.second->size(),0);
    return out;
}
} // namespace
void validate_geometry(const Geometry& g) {geometry(g);}
State zero_state(const Geometry& g) {
    geometry(g);State out;
    out.phase.resize(extent({g.batch,g.heads,g.rotary_pairs}));
    out.ssm.resize(extent({g.batch,g.heads,g.head_dim,g.state_dim}));
    out.k.resize(extent({g.batch,g.heads,g.state_dim}));out.v.resize(extent({g.batch,g.heads,g.head_dim}));return out;
}
Forward forward(const Inputs& input,const State& initial) {return evaluate(input,initial,false).result;}
Gradients backward(const Inputs& x,const State& initial,const std::vector<double>& dy,const State& final_gradient) {
    const auto tr=evaluate(x,initial,true);const auto& g=x.geometry;shape(dy,x.v.size(),"output adjoint");
    const State start=empty(initial)?zero_state(g):initial;
    State carry=empty(final_gradient)?zero_state(g):final_gradient;state_shape(carry,g);
    Gradients out{zeros_like(x),zero_state(g)};auto& dx=out.input;
    for (int b=0;b<g.batch;++b) for (int h=0;h<g.heads;++h) {
        const size_t bh=static_cast<size_t>(b)*g.heads+h;
        const int length=x.valid_lengths.empty()?g.sequence:x.valid_lengths[b];
        for (int t=length-1;t>=0;--t) {
            const size_t th=time_head(g,b,t,h),kn=th*g.state_dim,vp=th*g.head_dim;
            const double alpha=std::exp(x.adt[th]),lambda=sigmoid(x.trap[th]);
            const double beta=(1-lambda)*x.dt[th]*alpha,gamma=lambda*x.dt[th];
            std::vector<double> dq(g.state_dim),dk(g.state_dim),dv(g.head_dim),pk(g.state_dim),pv(g.head_dim);
            for (int n=0;n<g.state_dim;++n) dk[n]=carry.k[bh*g.state_dim+n];
            for (int p=0;p<g.head_dim;++p) dv[p]=carry.v[bh*g.head_dim+p];
            double da=0,db=0,dc=0;
            for (int p=0;p<g.head_dim;++p) {
                double y=0;
                for (int n=0;n<g.state_dim;++n) y+=tr.history[history_index(g,b,h,t+1,p,n)]*tr.qr[kn+n];
                if (!x.d.empty()) y+=x.d[h]*x.v[vp+p];
                double gy=dy[vp+p];
                if (!x.z.empty()) {
                    const double s=sigmoid(x.z[vp+p]);
                    dx.z[vp+p]=gy*y*(s+x.z[vp+p]*s*(1-s));gy*=x.z[vp+p]*s;
                }
                if (!x.d.empty()) {dx.d[h]+=gy*x.v[vp+p];dv[p]+=gy*x.d[h];}
                const double vprev=t?x.v[(time_head(g,b,t-1,h))*g.head_dim+p]:start.v[bh*g.head_dim+p];
                for (int n=0;n<g.state_dim;++n) {
                    const size_t si=state_index(g,b,h,p,n);
                    const double hp=tr.history[history_index(g,b,h,t,p,n)];
                    const double kp=t?tr.kr[time_head(g,b,t-1,h)*g.state_dim+n]:start.k[bh*g.state_dim+n];
                    dq[n]+=gy*tr.history[history_index(g,b,h,t+1,p,n)];
                    const double gh=carry.ssm[si]+gy*tr.qr[kn+n];
                    da+=gh*hp;db+=gh*kp*vprev;dc+=gh*tr.kr[kn+n]*x.v[vp+p];
                    dk[n]+=gh*gamma*x.v[vp+p];dv[p]+=gh*gamma*tr.kr[kn+n];
                    pk[n]+=gh*beta*vprev;pv[p]+=gh*beta*kp;carry.ssm[si]=gh*alpha;
                }
            }
            dx.adt[th]=(da+db*(1-lambda)*x.dt[th])*alpha;
            dx.dt[th]=db*(1-lambda)*alpha+dc*lambda;
            dx.trap[th]=(-db*x.dt[th]*alpha+dc*x.dt[th])*lambda*(1-lambda);
            for (int n=0;n<g.state_dim;n+=2) {
                const int r=n/2;const double phase=r<g.rotary_pairs?tr.phase[th*g.rotary_pairs+r]:0;
                const double c=std::cos(phase),s=std::sin(phase);
                const size_t qi=qindex(g,b,t,h,n),bi=static_cast<size_t>(h)*g.state_dim+n;
                dx.q[qi]+=dq[n]*c+dq[n+1]*s;dx.q[qi+1]+=-dq[n]*s+dq[n+1]*c;
                dx.k[qi]+=dk[n]*c+dk[n+1]*s;dx.k[qi+1]+=-dk[n]*s+dk[n+1]*c;
                dx.q_bias[bi]+=dq[n]*c+dq[n+1]*s;dx.q_bias[bi+1]+=-dq[n]*s+dq[n+1]*c;
                dx.k_bias[bi]+=dk[n]*c+dk[n+1]*s;dx.k_bias[bi+1]+=-dk[n]*s+dk[n+1]*c;
                if (r<g.rotary_pairs) {
                    double& gp=carry.phase[bh*g.rotary_pairs+r];
                    gp+=-dq[n]*tr.qr[kn+n+1]+dq[n+1]*tr.qr[kn+n]-dk[n]*tr.kr[kn+n+1]+dk[n+1]*tr.kr[kn+n];
                    const double angle=std::tanh(x.angles[th*g.rotary_pairs+r]);
                    dx.angles[th*g.rotary_pairs+r]=gp*pi*(1-angle*angle)*x.dt[th];
                    dx.dt[th]+=gp*pi*angle;
                }
            }
            for (int n=0;n<g.state_dim;++n) carry.k[bh*g.state_dim+n]=pk[n];
            for (int p=0;p<g.head_dim;++p) {dx.v[vp+p]=dv[p];carry.v[bh*g.head_dim+p]=pv[p];}
        }
    }
    out.initial_state=std::move(carry);return out;
}
} // namespace nsos::mamba3_reference

namespace nsos::mamba3_reference {
namespace {
using BS=mamba3_block::Shape;
void block_validate(const BlockInputs& x) {
    const auto& s=x.geometry;
    if(!mamba3_block::eligible(s)) throw std::invalid_argument("Mamba3 integral oracle geometry");
    shape(x.projection,std::size_t(s.batch)*s.sequence*s.width(),"block projection");
    shape(x.core,mamba3_block::Layout(s).total,"block core");
    if(!x.valid_lengths.empty()) {if(x.valid_lengths.size()!=std::size_t(s.batch)) throw std::invalid_argument("Mamba3 block prefixes");for(int n:x.valid_lengths) if(n<0||n>s.sequence) throw std::invalid_argument("Mamba3 block prefix range");}
}
void block_state_validate(const State& x,const BS& s) {
    shape(x.phase,std::size_t(s.batch)*s.heads*s.rotary_pairs,"block phase");
    shape(x.ssm,std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim,"block SSM");
    shape(x.k,std::size_t(s.batch)*s.heads*s.rank*s.state_dim,"block previous K");
    shape(x.v,std::size_t(s.batch)*s.heads*s.head_dim,"block raw previous V");
}
mamba3_block::State<const double> cstate(const State& s) {return {s.phase.data(),s.ssm.data(),s.k.data(),s.v.data()};}
mamba3_block::State<double> state(State& s) {return {s.phase.data(),s.ssm.data(),s.k.data(),s.v.data()};}
struct BlockTrace {Forward result;std::vector<double> history,q,k,phase,readout;};
BlockTrace block_evaluate(const BlockInputs& x,const State& initial) {
    block_validate(x);const auto& s=x.geometry;const auto start=empty(initial)?block_zero_state(s):initial;block_state_validate(start,s);
    BlockTrace tr;tr.result.final_state=block_zero_state(s);tr.result.output.resize(std::size_t(s.batch)*s.sequence*s.inner());
    tr.history.resize(mamba3_block::history_size(s));tr.q.resize(mamba3_block::rotation_size(s));tr.k.resize(tr.q.size());
    tr.phase.resize(mamba3_block::phase_size(s));tr.readout.resize(mamba3_block::readout_size(s));
    const auto valid=x.valid_lengths.empty()?std::vector<int>(s.batch,s.sequence):x.valid_lengths;std::vector<double> status(s.batch);
    for(int b=0;b<s.batch;++b) mamba3_block::detail::forward_batch(s,mamba3_block::Layout(s),b,x.projection.data(),x.core.data(),valid.data(),cstate(start),state(tr.result.final_state),{tr.history.data(),tr.q.data(),tr.k.data(),tr.phase.data(),tr.readout.data()},tr.result.output.data(),status.data());
    for(double code:status) if(code) throw std::runtime_error("Mamba3 integral FP64 oracle invalid/nonfinite forward");return tr;
}
}
State block_zero_state(const BS& s) {
    if(!mamba3_block::eligible(s)) throw std::invalid_argument("Mamba3 block state geometry");
    State out;const auto bh=std::size_t(s.batch)*s.heads;
    out.phase.resize(bh*s.rotary_pairs);out.ssm.resize(bh*s.head_dim*s.state_dim);out.k.resize(bh*s.rank*s.state_dim);out.v.resize(bh*s.head_dim);return out;
}
Forward block_forward(const BlockInputs& x,const State& initial) {return block_evaluate(x,initial).result;}
BlockGradients block_backward(const BlockInputs& x,const State& initial,const std::vector<double>& dy,const State& final_seed) {
    auto tr=block_evaluate(x,initial);const auto& s=x.geometry;shape(dy,tr.result.output.size(),"block dy");
    const auto start=empty(initial)?block_zero_state(s):initial,seed=empty(final_seed)?block_zero_state(s):final_seed;block_state_validate(seed,s);
    BlockGradients out;out.projection.resize(x.projection.size());out.core.resize(x.core.size());out.initial_state=block_zero_state(s);
    std::vector<double> partial(std::size_t(s.batch)*x.core.size()),scratch(std::size_t(s.batch)*mamba3_block::scratch_per_head(s)),status(s.batch);
    const auto valid=x.valid_lengths.empty()?std::vector<int>(s.batch,s.sequence):x.valid_lengths;
    for(int b=0;b<s.batch;++b) mamba3_block::detail::backward_batch(s,mamba3_block::Layout(s),b,x.projection.data(),x.core.data(),valid.data(),cstate(start),{tr.history.data(),tr.q.data(),tr.k.data(),tr.phase.data(),tr.readout.data()},dy.data(),cstate(seed),out.projection.data(),partial.data()+std::size_t(b)*out.core.size(),state(out.initial_state),scratch.data()+std::size_t(b)*mamba3_block::scratch_per_head(s),status.data());
    for(double code:status) if(code) throw std::runtime_error("Mamba3 integral FP64 oracle invalid/nonfinite VJP");
    for(int b=0;b<s.batch;++b) for(std::size_t i=0;i<out.core.size();++i) out.core[i]+=partial[std::size_t(b)*out.core.size()+i];return out;
}
}
