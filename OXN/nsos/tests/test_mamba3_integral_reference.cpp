#include "mamba3_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
namespace ref=nsos::mamba3_reference;
namespace mb=nsos::mamba3_block;
namespace {
void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
void close(double a,double b,const char* name,double tol=2e-7) {if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tol*(1+std::abs(b))) throw std::runtime_error(std::string(name)+" got="+std::to_string(a)+" expected="+std::to_string(b));}
void fill(std::vector<double>& v,double scale=.2,double bias=0) {for(std::size_t i=0;i<v.size();++i) v[i]=bias+scale*std::sin(.113*(i+1));}
double sg(double x) {return 1/(1+std::exp(-x));}
ref::BlockInputs fixture(bool mimo,bool norm,int N=8,int R=2) {
    mb::Shape s{2,3,4,2,1,2,N,mimo?R:1,N/4,mimo,norm,1e-5,1e-4};ref::BlockInputs x{s};
    x.projection.resize(std::size_t(s.batch)*s.sequence*s.width());fill(x.projection);
    x.core.resize(mb::Layout(s).total);fill(x.core,.05,1);mb::Layout l(s);for(int h=0;h<s.heads;++h) x.core[l.dt+h]=-2.;return x;
}
double dot(const std::vector<double>& a,const std::vector<double>& b) {require(a.size()==b.size(),"dot shape");double v=0;for(std::size_t i=0;i<a.size();++i) v+=a[i]*b[i];return v;}
double objective(const ref::BlockInputs& x,const ref::State& initial,const std::vector<double>& dy,const ref::State& seed) {
    auto f=ref::block_forward(x,initial);return dot(f.output,dy)+dot(f.final_state.phase,seed.phase)+dot(f.final_state.ssm,seed.ssm)+dot(f.final_state.k,seed.k)+dot(f.final_state.v,seed.v);
}
// Independent O(S^2) expansion of the trapezoidal recurrence. Does not call
// the production math header, its forward, or its normalization helpers.
std::vector<double> quadratic(const ref::BlockInputs& x,const ref::State& initial) {
    auto s=x.geometry;mb::Layout l(s);const int I=s.inner(),W=s.width(),N=s.state_dim,P=s.head_dim,R=s.rank,H=s.heads,G=s.groups,A=s.rotary_pairs;
    std::vector<double> out(std::size_t(s.batch)*s.sequence*I);
    for(int b=0;b<s.batch;++b) for(int h=0;h<H;++h) {
        const int L=x.valid_lengths.empty()?s.sequence:x.valid_lengths[b];const auto bh=std::size_t(b)*H+h;
        std::vector<double> q(L*R*N),k(q.size()),v(L*R*P),alpha(L),beta(L),gamma(L),phase(A);
        for(int j=0;j<A;++j) phase[j]=initial.phase[bh*A+j];
        for(int t=0;t<L;++t) {
            const double* raw=x.projection.data()+(std::size_t(b)*s.sequence+t)*W;
            const double ra=raw[2*I+2*R*G*N+H+h],a=-std::max(ra>=0?1+ra:1/(1-ra),s.a_floor),rd=raw[2*I+2*R*G*N+h]+x.core[l.dt+h],dt=std::log1p(std::exp(rd)),lam=sg(raw[2*I+2*R*G*N+2*H+h]);
            alpha[t]=std::exp(a*dt);beta[t]=(1-lam)*dt*alpha[t];gamma[t]=lam*dt;
            for(int j=0;j<A;++j) phase[j]+=3.14159265358979323846*std::tanh(raw[W-A+j])*dt;
            for(int r=0;r<R;++r) for(int kind=0;kind<2;++kind) {
                auto& dest=kind?k:q;const double* bc=raw+2*I+(kind?0:R*G*N)+(r*G+h/(H/G))*N;double sq=0;for(int n=0;n<N;++n) sq+=bc[n]*bc[n];
                for(int n=0;n<N;++n) dest[(t*R+r)*N+n]=bc[n]/std::sqrt(sq/N+s.norm_eps)*x.core[(kind?l.bnorm:l.cnorm)+n]+x.core[(kind?l.bbias:l.cbias)+(h*R+r)*N+n];
                for(int j=0;j<A;++j) {const int n=s.mimo?j:2*j,m=s.mimo?j+N/2:2*j+1;const auto i=(t*R+r)*N;const double u=dest[i+n],w=dest[i+m],c=std::cos(phase[j]),sn=std::sin(phase[j]);dest[i+n]=c*u-sn*w;dest[i+m]=sn*u+c*w;}
            }
            for(int r=0;r<R;++r) for(int p=0;p<P;++p) v[(t*R+r)*P+p]=raw[I+h*P+p]*(s.mimo?x.core[l.x+(h*R+r)*P+p]:1);
        }
        for(int t=0;t<L;++t) {
            std::vector<double> state(P*N),y(R*P);double decay=1;for(int i=0;i<=t;++i) decay*=alpha[i];
            for(int p=0;p<P;++p) for(int n=0;n<N;++n) {
                double cell=decay*initial.ssm[bh*P*N+p*N+n];
                for(int i=0;i<=t;++i) {double tail=1;for(int j=i+1;j<=t;++j) tail*=alpha[j];
                    for(int r=0;r<R;++r) {const double pk=i?k[((i-1)*R+r)*N+n]:initial.k[(bh*R+r)*N+n],pv=i?v[((i-1)*R+r)*P+p]:initial.v[bh*P+p]*(s.mimo?x.core[l.x+(h*R+r)*P+p]:1);
                        cell+=tail*(beta[i]*pk*pv+gamma[i]*k[(i*R+r)*N+n]*v[(i*R+r)*P+p]);}}
                state[p*N+n]=cell;
            }
            const auto row=std::size_t(b)*s.sequence+t;const double* raw=x.projection.data()+row*W;
            for(int r=0;r<R;++r) {
                double sq=0;for(int p=0;p<P;++p) {double yy=x.core[l.d+h]*v[(t*R+r)*P+p];for(int n=0;n<N;++n) yy+=state[p*N+n]*q[(t*R+r)*N+n];y[r*P+p]=yy;sq+=yy*yy;}
                for(int p=0;p<P;++p) {const int rp=(h*R+r)*P+p,hp=h*P+p;const double z=raw[hp]*(s.mimo?x.core[l.z+rp]:1);out[row*I+hp]+=y[r*P+p]*(s.out_norm?x.core[l.norm+hp]/std::sqrt(sq/P+s.norm_eps):1)*z*sg(z)*(s.mimo?x.core[l.o+rp]:1);}
            }
        }
    }return out;
}
void run(bool mimo,bool norm) {
    auto x=fixture(mimo,norm);x.valid_lengths={3,1};auto initial=ref::block_zero_state(x.geometry),seed=initial;
    for(auto* v:{&initial.phase,&initial.ssm,&initial.k,&initial.v}) fill(*v,.01,.07);for(auto* v:{&seed.phase,&seed.ssm,&seed.k,&seed.v}) fill(*v,.013);
    auto f=ref::block_forward(x,initial);auto expected=quadratic(x,initial);for(std::size_t i=0;i<expected.size();++i) close(f.output[i],expected[i],"quadratic",2e-10);
    std::vector<double> dy(f.output.size());fill(dy,.07);auto grad=ref::block_backward(x,initial,dy,seed);
    const double eps=2e-6;
    for(auto pair:{std::pair{&x.projection,&grad.projection},std::pair{&x.core,&grad.core},std::pair{&initial.phase,&grad.initial_state.phase},std::pair{&initial.ssm,&grad.initial_state.ssm},std::pair{&initial.k,&grad.initial_state.k},std::pair{&initial.v,&grad.initial_state.v}}) {
        for(std::size_t i=0;i<pair.first->size();++i) {double saved=(*pair.first)[i];(*pair.first)[i]=saved+eps;double plus=objective(x,initial,dy,seed);(*pair.first)[i]=saved-eps;double minus=objective(x,initial,dy,seed);(*pair.first)[i]=saved;close((*pair.second)[i],(plus-minus)/(2*eps),"finite difference",2e-6);}
    }
}
}
int main() {try {for(bool mimo:{false,true}) for(bool norm:{false,true}) run(mimo,norm);
    for(bool mimo:{false,true}) for(int rank:{1,4,8}) {auto x=fixture(mimo,true,128,rank);auto f=ref::block_forward(x);for(double y:f.output) require(std::isfinite(y),"N128/rank8 forward");std::vector<double> dy(f.output.size(),.001);ref::block_backward(x,{},dy);}
    auto x=fixture(true,true);x.valid_lengths={0,0};std::fill(x.projection.begin(),x.projection.end(),std::numeric_limits<double>::quiet_NaN());auto initial=ref::block_zero_state(x.geometry);fill(initial.ssm);auto f=ref::block_forward(x,initial);require(f.final_state.ssm==initial.ssm,"zero prefix state");for(double y:f.output) require(y==0,"zero prefix output");
    std::cout<<"Mamba3 integral FP64: quadratic + exhaustive finite differences + N128/ranks/masks PASS\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
