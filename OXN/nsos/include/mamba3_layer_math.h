#pragma once
#include "cuda/mamba3_layer_kernels.cuh"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NSOS_M3_HD __host__ __device__
#else
#define NSOS_M3_HD
#endif
namespace nsos::mamba3_block::detail {
constexpr double pi=3.1415926535897932384626433832795,tau=2*pi;
template<class T> NSOS_M3_HD T sigmoid(T x) {return x>=0?T(1)/(T(1)+exp(-x)):exp(x)/(T(1)+exp(x));}
template<class T> NSOS_M3_HD T softplus(T x) {return (x>0?x:T(0))+log1p(exp(-fabs(x)));}
template<class T> NSOS_M3_HD T heavy(T x) {return x>=0?T(1)+x:T(1)/(T(1)-x);}
template<class T> NSOS_M3_HD bool finite(T x) {return x==x&&fabs(x)<=std::numeric_limits<T>::max();}
template<class T> NSOS_M3_HD T value(const T* p,std::size_t i) {return p?p[i]:T(0);}
template<class T> NSOS_M3_HD void copy(T* out,const T* in,std::size_t n) {for(std::size_t i=0;i<n;++i) out[i]=value(in,i);}
template<class T> NSOS_M3_HD T inverse(const T* p,int n,double eps) {double v=0;for(int i=0;i<n;++i) v+=double(p[i])*p[i];return T(1/sqrt(v/n+eps));}
NSOS_M3_HD inline int coord(const Shape& s,int pair,bool second) {return s.mimo?pair+(second?s.state_dim/2:0):2*pair+(second?1:0);}
NSOS_M3_HD inline std::size_t state_offset(const Shape& s,int b,int h) {return std::size_t(b)*s.heads+h;}
// All forward outputs/traces are initialized by their owners. One serial owner
// per batch avoids cross-head/group gradient races and guarantees fixed order.
template<class T> NSOS_M3_HD void forward_batch(const Shape& s,const Layout& l,int b,const T* proj,const T* w,const int* valid,
    State<const T> initial,State<T> final,Trace<T> tr,T* out,T* status) {
    const int N=s.state_dim,P=s.head_dim,R=s.rank,A=s.rotary_pairs,H=s.heads,G=s.groups,I=H*P,W=2*I+2*R*G*N+3*H+A;
    const int len=valid[b];if(len<0||len>s.sequence) {status[b]=1;return;}
    for(std::size_t i=0;i<l.total;++i) if(!finite(w[i])) {status[b]=2;return;}
    for(int h=0;h<H;++h) {
        const auto bh=state_offset(s,b,h),hs=bh*P*N,ks=bh*R*N,vs=bh*P,ps=bh*A;
        copy(final.phase+ps,initial.phase?initial.phase+ps:nullptr,A);
        copy(final.ssm+hs,initial.ssm?initial.ssm+hs:nullptr,P*N);
        copy(final.k+ks,initial.k?initial.k+ks:nullptr,R*N);
        copy(final.v+vs,initial.v?initial.v+vs:nullptr,P);
        for(int n=0;n<P*N;++n) if(!finite(final.ssm[hs+n])) {status[b]=2;return;}
        for(int n=0;n<R*N;++n) if(!finite(final.k[ks+n])) {status[b]=2;return;}
        for(int p=0;p<P;++p) if(!finite(final.v[vs+p])) {status[b]=2;return;}
        for(int a=0;a<A;++a) if(!finite(final.phase[ps+a])) {status[b]=2;return;}
        T* hist=tr.history+bh*(s.sequence+1)*P*N;copy(hist,final.ssm+hs,P*N);
        for(int t=0;t<len;++t) {
            const auto row=(std::size_t(b)*s.sequence+t);const T* raw=proj+row*W;
            for(int i=0;i<W;++i) if(!finite(raw[i])) {status[b]=2;return;}
            const T act=heavy(raw[2*I+2*R*G*N+H+h]),a=-(act>T(s.a_floor)?act:T(s.a_floor));
            const T dt=softplus(raw[2*I+2*R*G*N+h]+w[l.dt+h]),alpha=exp(a*dt),lam=sigmoid(raw[2*I+2*R*G*N+2*H+h]);
            const T beta=(1-lam)*dt*alpha,gamma=lam*dt;
            for(int j=0;j<A;++j) {T phase=final.phase[ps+j]+T(pi)*tanh(raw[W-A+j])*dt;phase-=T(tau)*floor(phase/T(tau));final.phase[ps+j]=phase;tr.phase[(bh*s.sequence+t)*A+j]=phase;}
            T* qr=tr.q+(bh*s.sequence+t)*R*N;T* kr=tr.k+(bh*s.sequence+t)*R*N;
            for(int r=0;r<R;++r) for(int kind=0;kind<2;++kind) {
                const T* bc=raw+2*I+(kind?0:R*G*N)+(r*G+h/(H/G))*N;
                const auto norm=kind?l.bnorm:l.cnorm,bias=(kind?l.bbias:l.cbias)+(h*R+r)*N;T* rot=(kind?kr:qr)+r*N;
                const T inv=inverse(bc,N,s.norm_eps);for(int n=0;n<N;++n) rot[n]=bc[n]*inv*w[norm+n]+w[bias+n];
                for(int j=0;j<A;++j) {const int n=coord(s,j,false),m=coord(s,j,true);const T c=cos(final.phase[ps+j]),sn=sin(final.phase[ps+j]),u=rot[n],v=rot[m];rot[n]=u*c-v*sn;rot[m]=u*sn+v*c;}
            }
            for(int p=0;p<P;++p) for(int n=0;n<N;++n) {
                T prev=0,curr=0;for(int r=0;r<R;++r) {const T px=s.mimo?w[l.x+(h*R+r)*P+p]:T(1);prev+=final.k[ks+r*N+n]*final.v[vs+p]*px;curr+=kr[r*N+n]*raw[I+h*P+p]*px;}
                T& cell=final.ssm[hs+p*N+n];cell=alpha*cell+beta*prev+gamma*curr;hist[(t+1)*P*N+p*N+n]=cell;
                if(!finite(cell)) {status[b]=3;return;}
            }
            T* y=tr.readout+(bh*s.sequence+t)*R*P;
            for(int r=0;r<R;++r) {
                for(int p=0;p<P;++p) {T acc=0;for(int n=0;n<N;++n) acc+=final.ssm[hs+p*N+n]*qr[r*N+n];const T px=s.mimo?w[l.x+(h*R+r)*P+p]:T(1);y[r*P+p]=acc+w[l.d+h]*raw[I+h*P+p]*px;}
                const T inv=s.out_norm?inverse(y+r*P,P,s.norm_eps):T(1);
                for(int p=0;p<P;++p) {const auto hp=h*P+p,rp=(h*R+r)*P+p;const T pz=s.mimo?w[l.z+rp]:T(1),po=s.mimo?w[l.o+rp]:T(1),z=raw[hp]*pz;
                    const T norm=s.out_norm?w[l.norm+hp]:T(1);out[row*I+hp]+=y[r*P+p]*inv*norm*z*sigmoid(z)*po;
                    if(!finite(out[row*I+hp])) {status[b]=3;return;}
                }
            }
            copy(final.k+ks,kr,R*N);for(int p=0;p<P;++p) final.v[vs+p]=raw[I+h*P+p];
        }
    }
}
template<class T> NSOS_M3_HD void backward_batch(const Shape& s,const Layout& l,int b,const T* proj,const T* w,const int* valid,
    State<const T> initial,Trace<const T> tr,const T* dy,State<const T> seed,T* dx,T* dp,State<T> di,T* scratch,T* status) {
    if(status[b]) return;
    const int N=s.state_dim,P=s.head_dim,R=s.rank,A=s.rotary_pairs,H=s.heads,G=s.groups,I=H*P,W=2*I+2*R*G*N+3*H+A,len=valid[b];
    for(int h=0;h<H;++h) {
        const auto bh=state_offset(s,b,h),hs=bh*P*N,ks=bh*R*N,vs=bh*P,ps=bh*A;
        T* dh=scratch;T* ck=dh+P*N;T* dq=ck+R*N;T* dk=dq+R*N;T* cv=dk+R*N;T* gy=cv+R*P;T* gp=gy+R*P;
        copy(dh,seed.ssm?seed.ssm+hs:nullptr,P*N);copy(ck,seed.k?seed.k+ks:nullptr,R*N);copy(gp,seed.phase?seed.phase+ps:nullptr,A);
        // cv stores adjoint to RAW previous value only in its first P entries.
        copy(cv,seed.v?seed.v+vs:nullptr,P);for(int i=0;i<P*N;++i) if(!finite(dh[i])) {status[b]=2;return;}
        for(int i=0;i<R*N;++i) if(!finite(ck[i])) {status[b]=2;return;}for(int i=0;i<P;++i) if(!finite(cv[i])) {status[b]=2;return;}for(int i=0;i<A;++i) if(!finite(gp[i])) {status[b]=2;return;}
        const T* hist=tr.history+bh*(s.sequence+1)*P*N;
        for(int t=len-1;t>=0;--t) {
            const auto row=std::size_t(b)*s.sequence+t;const T* raw=proj+row*W;T* dr=dx+row*W;
            const T* qr=tr.q+(bh*s.sequence+t)*R*N;const T* kr=tr.k+(bh*s.sequence+t)*R*N;
            const T* pk=t?tr.k+(bh*s.sequence+t-1)*R*N:(initial.k?initial.k+ks:nullptr);
            const T* prev=t?proj+(row-1)*W+I+h*P:(initial.v?initial.v+vs:nullptr);
            const T* y=tr.readout+(bh*s.sequence+t)*R*P;const T* phase=tr.phase+(bh*s.sequence+t)*A;
            const T act=heavy(raw[2*I+2*R*G*N+H+h]),a=-(act>T(s.a_floor)?act:T(s.a_floor)),dt=softplus(raw[2*I+2*R*G*N+h]+w[l.dt+h]);
            const T alpha=exp(a*dt),lam=sigmoid(raw[2*I+2*R*G*N+2*H+h]),beta=(1-lam)*dt*alpha,gamma=lam*dt;
            for(int p=0;p<P;++p) {dr[I+h*P+p]+=cv[p];cv[p]=0;}
            copy(dk,ck,R*N);for(int i=0;i<R*N;++i) {dq[i]=0;ck[i]=0;}
            for(int r=0;r<R;++r) {
                const T inv=s.out_norm?inverse(y+r*P,P,s.norm_eps):T(1);double cross=0;
                for(int p=0;p<P;++p) {
                    const int hp=h*P+p,rp=(h*R+r)*P+p;const T gz=raw[hp]*(s.mimo?w[l.z+rp]:T(1)),sg=sigmoid(gz),gate=gz*sg,norm=s.out_norm?w[l.norm+hp]:T(1),po=s.mimo?w[l.o+rp]:T(1);
                    const T upstream=dy[row*I+hp];if(!finite(upstream)) {status[b]=2;return;}
                    const T dgate=upstream*po*y[r*P+p]*inv*norm*sg*(1+gz*(1-sg));
                    dr[hp]+=dgate*(s.mimo?w[l.z+rp]:T(1));if(s.mimo) {dp[l.z+rp]+=dgate*raw[hp];dp[l.o+rp]+=upstream*y[r*P+p]*inv*norm*gate;}
                    if(s.out_norm) dp[l.norm+hp]+=upstream*po*gate*y[r*P+p]*inv;
                    gy[r*P+p]=upstream*po*gate*norm;cross+=double(gy[r*P+p])*y[r*P+p];
                }
                for(int p=0;p<P;++p) {
                    const int rp=(h*R+r)*P+p;const T px=s.mimo?w[l.x+rp]:T(1);
                    const T adj=s.out_norm?T(double(inv)*(double(gy[r*P+p])-double(y[r*P+p])*double(inv)*inv*cross/P)):gy[r*P+p];gy[r*P+p]=adj;
                    dp[l.d+h]+=adj*raw[I+h*P+p]*px;dr[I+h*P+p]+=adj*w[l.d+h]*px;
                    if(s.mimo) dp[l.x+rp]+=adj*w[l.d+h]*raw[I+h*P+p];
                    for(int n=0;n<N;++n) dq[r*N+n]+=adj*hist[(t+1)*P*N+p*N+n];
                }
            }
            T da=0,db=0,dc=0;
            for(int p=0;p<P;++p) for(int n=0;n<N;++n) {
                const int cell=p*N+n;T gh=dh[cell];for(int r=0;r<R;++r) gh+=gy[r*P+p]*qr[r*N+n];da+=gh*hist[t*P*N+cell];
                for(int r=0;r<R;++r) {
                    const int rp=(h*R+r)*P+p;const T px=s.mimo?w[l.x+rp]:T(1),vp=value(prev,p),kp=value(pk,r*N+n),v=raw[I+h*P+p];
                    db+=gh*kp*vp*px;dc+=gh*kr[r*N+n]*v*px;
                    dk[r*N+n]+=gh*gamma*v*px;dr[I+h*P+p]+=gh*gamma*kr[r*N+n]*px;
                    ck[r*N+n]+=gh*beta*vp*px;cv[p]+=gh*beta*kp*px;
                    if(s.mimo) dp[l.x+rp]+=gh*(gamma*kr[r*N+n]*v+beta*kp*vp);
                }
                dh[cell]=gh*alpha;
            }
            T dadt=(da+db*(1-lam)*dt)*alpha,ddt=db*(1-lam)*alpha+dc*lam;
            dr[2*I+2*R*G*N+2*H+h]+=(-db*dt*alpha+dc*dt)*lam*(1-lam);
            for(int r=0;r<R;++r) {
                for(int j=0;j<A;++j) {
                    const int n=coord(s,j,false),m=coord(s,j,true);const T c=cos(phase[j]),sn=sin(phase[j]);
                    gp[j]+=-dq[r*N+n]*qr[r*N+m]+dq[r*N+m]*qr[r*N+n]-dk[r*N+n]*kr[r*N+m]+dk[r*N+m]*kr[r*N+n];
                    const T qn=dq[r*N+n],qm=dq[r*N+m],kn=dk[r*N+n],km=dk[r*N+m];
                    dq[r*N+n]=qn*c+qm*sn;dq[r*N+m]=-qn*sn+qm*c;dk[r*N+n]=kn*c+km*sn;dk[r*N+m]=-kn*sn+km*c;
                }
                for(int kind=0;kind<2;++kind) {
                    const int off=2*I+(kind?0:R*G*N)+(r*G+h/(H/G))*N;const T* bc=raw+off;T* dbc=dr+off;const T* adj=(kind?dk:dq)+r*N;
                    const auto norm=kind?l.bnorm:l.cnorm,bias=(kind?l.bbias:l.cbias)+(h*R+r)*N;
                    // FP64 radial reduction even for the FP32 baseline.
                    double sq=0,cross=0;for(int n=0;n<N;++n) {sq+=double(bc[n])*bc[n];cross+=double(bc[n])*w[norm+n]*adj[n];}
                    const double inv=1/sqrt(sq/N+s.norm_eps);
                    for(int n=0;n<N;++n) {dbc[n]+=T(inv*(double(w[norm+n])*adj[n]-double(bc[n])*inv*inv*cross/N));dp[norm+n]+=T(double(adj[n])*bc[n]*inv);dp[bias+n]+=adj[n];}
                }
            }
            for(int j=0;j<A;++j) {const T angle=tanh(raw[W-A+j]);dr[W-A+j]+=gp[j]*T(pi)*(1-angle*angle)*dt;ddt+=gp[j]*T(pi)*angle;}
            const T gdt=(ddt+a*dadt)*sigmoid(raw[2*I+2*R*G*N+h]+w[l.dt+h]);dr[2*I+2*R*G*N+h]+=gdt;dp[l.dt+h]+=gdt;
            dr[2*I+2*R*G*N+H+h]+=act>T(s.a_floor)?-dadt*dt*(raw[2*I+2*R*G*N+H+h]>=0?T(1):act*act):T(0);
        }
        copy(di.phase+ps,gp,A);copy(di.ssm+hs,dh,P*N);copy(di.k+ks,ck,R*N);copy(di.v+vs,cv,P);
        for(int i=0;i<P*N;++i) if(!finite(dh[i])) status[b]=3;
        for(int i=0;i<R*N;++i) if(!finite(ck[i])) status[b]=3;for(int i=0;i<P;++i) if(!finite(cv[i])) status[b]=3;for(int i=0;i<A;++i) if(!finite(gp[i])) status[b]=3;
    }
    for(std::size_t i=0;i<l.total;++i) if(!finite(dp[i])) status[b]=3;
    for(std::size_t i=0;i<std::size_t(len)*W;++i) if(!finite(dx[std::size_t(b)*s.sequence*W+i])) status[b]=3;
}
}
#undef NSOS_M3_HD
