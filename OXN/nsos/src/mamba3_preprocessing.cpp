#include "mamba3_preprocessing.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nsos::mamba3_preprocessing {
namespace {
void shape(const std::vector<double>& v,size_t n,const char* field) {
    if (v.size()!=n) throw std::invalid_argument(std::string("Mamba3 preprocessing shape: ")+field);
}
void validate(const Inputs& x) {
    // Reuse the oracle's overflow/geometry checks before computing extents.
    mamba3_reference::validate_geometry(x.geometry);
    const auto g=x.geometry;
    const size_t rows=static_cast<size_t>(g.batch)*g.sequence;
    shape(x.q,rows*g.groups*g.state_dim,"Q");shape(x.k,x.q.size(),"K");
    shape(x.v,rows*g.heads*g.head_dim,"V");if (!x.z.empty()) shape(x.z,x.v.size(),"Z");
    shape(x.raw_a,rows*g.heads,"raw A");shape(x.raw_dt,x.raw_a.size(),"raw DT");shape(x.trap,x.raw_a.size(),"Trap");
    shape(x.angles,rows*g.rotary_pairs,"shared angles");shape(x.q_norm,g.state_dim,"Q norm");shape(x.k_norm,g.state_dim,"K norm");
    shape(x.dt_bias,g.heads,"DT bias");shape(x.q_bias,static_cast<size_t>(g.heads)*g.state_dim,"Q bias");shape(x.k_bias,x.q_bias.size(),"K bias");
    if (!x.d.empty()) shape(x.d,g.heads,"D");
    if (!std::isfinite(x.norm_eps)||x.norm_eps<=0||!std::isfinite(x.a_floor)||x.a_floor<=0)
        throw std::invalid_argument("Mamba3 preprocessing invalid fixed coefficients");
    if (!x.valid_lengths.empty()) {
        if (x.valid_lengths.size()!=static_cast<size_t>(g.batch)) throw std::invalid_argument("Mamba3 preprocessing prefix shape");
        for (int n:x.valid_lengths) if (n<0||n>g.sequence) throw std::invalid_argument("Mamba3 preprocessing invalid prefix");
    }
}
double softplus(double x) {return std::max(x,0.0)+std::log1p(std::exp(-std::abs(x)));}
double sigmoid(double x) {return x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x));}
double heavy(double x) {return x>=0?1+x:1/(1-x);}
void zero(std::vector<double>& v) {std::fill(v.begin(),v.end(),0.0);}
}
mamba3_reference::Inputs forward(const Inputs& x) {
    validate(x);const auto g=x.geometry;mamba3_reference::Inputs out;
    out.geometry=g;out.valid_lengths=x.valid_lengths;
    out.q.resize(x.q.size());out.k.resize(x.k.size());out.v.resize(x.v.size());out.z.resize(x.z.size());
    out.adt.resize(x.raw_a.size());out.dt.resize(x.raw_dt.size());out.trap.resize(x.trap.size());
    out.angles.resize(x.raw_a.size()*g.rotary_pairs);out.q_bias=x.q_bias;out.k_bias=x.k_bias;out.d=x.d;
    for (int b=0;b<g.batch;++b) {
        const int length=x.valid_lengths.empty()?g.sequence:x.valid_lengths[b];
        for (int t=0;t<length;++t) {
            const size_t row=static_cast<size_t>(b)*g.sequence+t;
            for (int group=0;group<g.groups;++group) {
                const size_t pos=(row*g.groups+group)*g.state_dim;
                for (int kind=0;kind<2;++kind) {
                    const auto& raw=kind?x.k:x.q;const auto& weight=kind?x.k_norm:x.q_norm;auto& norm=kind?out.k:out.q;
                    double square=0;for (int n=0;n<g.state_dim;++n) square+=raw[pos+n]*raw[pos+n];
                    const double inverse=1/std::sqrt(square/g.state_dim+x.norm_eps);
                    for (int n=0;n<g.state_dim;++n) norm[pos+n]=raw[pos+n]*inverse*weight[n];
                }
            }
            for (int h=0;h<g.heads;++h) {
                const size_t th=row*g.heads+h;
                const double a=-std::max(heavy(x.raw_a[th]),x.a_floor);
                out.dt[th]=softplus(x.raw_dt[th]+x.dt_bias[h]);out.adt[th]=a*out.dt[th];out.trap[th]=x.trap[th];
                for (int r=0;r<g.rotary_pairs;++r) out.angles[th*g.rotary_pairs+r]=x.angles[row*g.rotary_pairs+r];
                for (int p=0;p<g.head_dim;++p) {
                    out.v[th*g.head_dim+p]=x.v[th*g.head_dim+p];
                    if (!x.z.empty()) out.z[th*g.head_dim+p]=x.z[th*g.head_dim+p];
                }
            }
        }
    }
    return out;
}
Inputs backward(const Inputs& x,const mamba3_reference::Inputs& grad) {
    validate(x);const auto g=x.geometry;
    if (grad.geometry.batch!=g.batch||grad.geometry.sequence!=g.sequence||grad.geometry.heads!=g.heads||grad.geometry.groups!=g.groups||
        grad.geometry.head_dim!=g.head_dim||grad.geometry.state_dim!=g.state_dim||grad.geometry.rotary_pairs!=g.rotary_pairs||grad.valid_lengths!=x.valid_lengths)
        throw std::invalid_argument("Mamba3 preprocessing adjoint metadata mismatch");
    shape(grad.q,x.q.size(),"dQ");shape(grad.k,x.k.size(),"dK");shape(grad.v,x.v.size(),"dV");shape(grad.z,x.z.size(),"dZ");
    shape(grad.adt,x.raw_a.size(),"dADT");shape(grad.dt,x.raw_dt.size(),"dDT");shape(grad.trap,x.trap.size(),"dTrap");
    shape(grad.angles,x.raw_a.size()*g.rotary_pairs,"dAngles");shape(grad.q_bias,x.q_bias.size(),"dQ bias");shape(grad.k_bias,x.k_bias.size(),"dK bias");shape(grad.d,x.d.size(),"dD");
    Inputs out=x;
    for (auto* v:{&out.q,&out.k,&out.v,&out.z,&out.raw_a,&out.raw_dt,&out.trap,&out.angles,&out.q_norm,&out.k_norm,&out.dt_bias,&out.q_bias,&out.k_bias,&out.d}) zero(*v);
    out.q_bias=grad.q_bias;out.k_bias=grad.k_bias;out.d=grad.d;
    for (int b=0;b<g.batch;++b) {
        const int length=x.valid_lengths.empty()?g.sequence:x.valid_lengths[b];
        for (int t=0;t<length;++t) {
            const size_t row=static_cast<size_t>(b)*g.sequence+t;
            for (int group=0;group<g.groups;++group) {
                const size_t pos=(row*g.groups+group)*g.state_dim;
                for (int kind=0;kind<2;++kind) {
                    const auto& raw=kind?x.k:x.q;const auto& weight=kind?x.k_norm:x.q_norm;const auto& adj=kind?grad.k:grad.q;
                    auto& draw=kind?out.k:out.q;auto& dw=kind?out.k_norm:out.q_norm;
                    double square=0,cross=0;
                    for (int n=0;n<g.state_dim;++n) {square+=raw[pos+n]*raw[pos+n];cross+=raw[pos+n]*weight[n]*adj[pos+n];}
                    const double inverse=1/std::sqrt(square/g.state_dim+x.norm_eps);
                    for (int n=0;n<g.state_dim;++n) {
                        draw[pos+n]=inverse*weight[n]*adj[pos+n]-raw[pos+n]*inverse*inverse*inverse*cross/g.state_dim;
                        dw[n]+=adj[pos+n]*raw[pos+n]*inverse;
                    }
                }
            }
            for (int h=0;h<g.heads;++h) {
                const size_t th=row*g.heads+h;
                const double act=heavy(x.raw_a[th]),a=-std::max(act,x.a_floor),dt=softplus(x.raw_dt[th]+x.dt_bias[h]);
                out.raw_a[th]=act>x.a_floor?-grad.adt[th]*dt*(x.raw_a[th]>=0?1:act*act):0;
                out.raw_dt[th]=(grad.dt[th]+a*grad.adt[th])*sigmoid(x.raw_dt[th]+x.dt_bias[h]);
                out.dt_bias[h]+=out.raw_dt[th];out.trap[th]=grad.trap[th];
                for (int r=0;r<g.rotary_pairs;++r) out.angles[row*g.rotary_pairs+r]+=grad.angles[th*g.rotary_pairs+r];
                for (int p=0;p<g.head_dim;++p) {
                    out.v[th*g.head_dim+p]=grad.v[th*g.head_dim+p];
                    if (!x.z.empty()) out.z[th*g.head_dim+p]=grad.z[th*g.head_dim+p];
                }
            }
        }
    }
    return out;
}
}
