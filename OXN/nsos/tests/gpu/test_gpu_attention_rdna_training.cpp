#include "gpu_parity_common.h"
#include "cuda/attention_rdna_training.cuh"
#include "cuda/device_buffer.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace nsos;
namespace ar = nsos::attention_rdna;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
std::size_t index(const ar::Shape& s, int b, int i, int h, int heads, int d = 0) {
    return ((static_cast<std::size_t>(b)*s.sequence+i)*heads+h)*s.head_dim+d;
}
std::size_t row(const ar::Shape& s, int b, int i, int h) {
    return (static_cast<std::size_t>(b)*s.sequence+i)*s.query_heads+h;
}

// Independent lowp conversion, not rocWMMA or the provider's cast helpers.
double rounded(float x, ar::Precision precision) {
    if (!std::isfinite(x) || x == 0) return x;
    if (precision == ar::Precision::BF16) {
        std::uint32_t bits;
        std::memcpy(&bits,&x,sizeof(bits));
        bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        float y;
        std::memcpy(&y,&bits,sizeof(y));
        return y;
    }
    int exponent = 0;
    const double a = std::abs(static_cast<double>(x));
    std::frexp(a,&exponent);
    const double quantum = std::ldexp(1.0,std::max(-24,exponent-11));
    const double units = a / quantum, base = std::floor(units), rest = units-base;
    const double integer = base + (rest > 0.5 || (rest == 0.5 && std::fmod(base,2.0) == 1.0) ? 1.0 : 0.0);
    const double value = integer*quantum;
    return std::copysign(value > 65504.0 ? std::numeric_limits<double>::infinity() : value,x);
}
std::vector<double> effective(const std::vector<float>& x, ar::Precision p) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = rounded(x[i],p);
    return y;
}
struct Reference {
    std::vector<double> out, dq, dk, dv, max, inv, delta;
};

// Scalar/double, dense probabilities on HOST only. No provider helpers,
// online recurrence, tiled reduction or GPU GEMM is used by this oracle.
Reference reference(const ar::Shape& s, const std::vector<int>& valid,
    const std::vector<double>& q, const std::vector<double>& k,
    const std::vector<double>& v, const std::vector<float>& go) {
    Reference r;
    r.out.assign(q.size(),0); r.dq.assign(q.size(),0);
    r.dk.assign(k.size(),0); r.dv.assign(v.size(),0);
    r.max.assign(ar::row_elements(s),0); r.inv = r.max; r.delta = r.max;
    for (int b = 0; b < s.batch; ++b) for (int h = 0; h < s.query_heads; ++h) {
        const int kh = h/(s.query_heads/s.kv_heads);
        for (int i = 0; i < valid[b]; ++i) {
            const int begin = std::max(0,i-s.window+1), count = i-begin+1;
            std::vector<double> scores(count), p(count), dp(count);
            for (int j = begin; j <= i; ++j) {
                double score = 0;
                for (int d = 0; d < s.head_dim; ++d)
                    score += q[index(s,b,i,h,s.query_heads,d)] * k[index(s,b,j,kh,s.kv_heads,d)];
                scores[j-begin] = score*s.scale;
            }
            const double mx = *std::max_element(scores.begin(),scores.end());
            double sum = 0;
            for (int z = 0; z < count; ++z) { p[z] = std::exp(scores[z]-mx); sum += p[z]; }
            const auto ri = row(s,b,i,h);
            r.max[ri] = mx; r.inv[ri] = 1/sum;
            for (double& prob : p) prob /= sum;
            for (int j = begin; j <= i; ++j) for (int d = 0; d < s.head_dim; ++d)
                r.out[index(s,b,i,h,s.query_heads,d)] += p[j-begin]*v[index(s,b,j,kh,s.kv_heads,d)];
            double delta = 0;
            for (int j = begin; j <= i; ++j) {
                for (int d = 0; d < s.head_dim; ++d)
                    dp[j-begin] += go[index(s,b,i,h,s.query_heads,d)]*v[index(s,b,j,kh,s.kv_heads,d)];
                delta += p[j-begin]*dp[j-begin];
            }
            r.delta[ri] = delta;
            for (int j = begin; j <= i; ++j) {
                const double ds = p[j-begin]*(dp[j-begin]-delta)*s.scale;
                for (int d = 0; d < s.head_dim; ++d) {
                    const auto qi = index(s,b,i,h,s.query_heads,d), ki = index(s,b,j,kh,s.kv_heads,d);
                    r.dq[qi] += ds*k[ki]; r.dk[ki] += ds*q[qi]; r.dv[ki] += p[j-begin]*go[qi];
                }
            }
        }
    }
    return r;
}
void close(const std::vector<float>& got, const std::vector<double>& want,
    const char* label, double atol = 3e-5, double rtol = 3e-4) {
    require(got.size() == want.size(),"oracle size mismatch");
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!std::isfinite(got[i]) || std::abs(got[i]-want[i]) > atol+rtol*std::abs(want[i]))
            throw std::runtime_error(std::string(label)+" mismatch index="+std::to_string(i)+
                " actual="+std::to_string(got[i])+" expected="+std::to_string(want[i]));
}

#ifdef USE_CUDA
template<class T> struct Buffer {
    cuda_detail::DeviceBuffer<T> storage;
    std::size_t size;
    static constexpr T canary = static_cast<T>(123456);
    explicit Buffer(std::size_t n) : size(n) {
        require(storage.ensure(n+4) != nullptr,"device fixture allocation failed");
        const std::vector<T> initial(n+4,canary);
        upload_raw(initial.data(),initial.size());
    }
    T* get() { return storage.get(); }
    void upload_raw(const T* x, std::size_t n) {
        require(cudaMemcpy(get(),x,n*sizeof(T),cudaMemcpyHostToDevice) == cudaSuccess,"fixture upload failed");
    }
    void upload(const std::vector<T>& x) {
        require(x.size() == size,"fixture upload size mismatch"); upload_raw(x.data(),x.size());
    }
    std::vector<T> read() {
        std::vector<T> x(size+4);
        require(cudaMemcpy(x.data(),get(),x.size()*sizeof(T),cudaMemcpyDeviceToHost) == cudaSuccess,"fixture download failed");
        for (std::size_t i = size; i < x.size(); ++i) require(x[i] == canary,"provider overwrote tail canary");
        x.resize(size); return x;
    }
};
struct Fixture {
    ar::Shape s;
    std::vector<int> valid;
    std::vector<float> q,k,v,go;
    Buffer<float> qg,kg,vg,gg,out,mx,inv,delta,dq,dk,dv;
    Buffer<int> lengths,status;
    Fixture(ar::Shape shape, std::vector<int> prefixes) : s(shape), valid(std::move(prefixes)),
        q(ar::query_elements(s)), k(ar::kv_elements(s)), v(k.size()), go(q.size()),
        qg(q.size()),kg(k.size()),vg(k.size()),gg(go.size()),out(q.size()),
        mx(ar::row_elements(s)),inv(ar::row_elements(s)),delta(ar::row_elements(s)),
        dq(q.size()),dk(k.size()),dv(k.size()),lengths(s.batch),status(s.batch) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (int b = 0; b < s.batch; ++b) for (int i = 0; i < s.sequence; ++i) {
            for (int h = 0; h < s.query_heads; ++h) for (int d = 0; d < s.head_dim; ++d) {
                const auto x = index(s,b,i,h,s.query_heads,d);
                q[x] = i < valid[b] ? 0.51f*std::sin(0.071f*static_cast<float>(x+1)) : nan;
                go[x] = i < valid[b] ? 0.23f*std::cos(0.059f*static_cast<float>(x+2)) : nan;
            }
            for (int h = 0; h < s.kv_heads; ++h) for (int d = 0; d < s.head_dim; ++d) {
                const auto x = index(s,b,i,h,s.kv_heads,d);
                k[x] = i < valid[b] ? 0.43f*std::cos(0.087f*static_cast<float>(x+3)) : nan;
                v[x] = i < valid[b] ? 0.69f*std::sin(0.053f*static_cast<float>(x+4)) : nan;
            }
        }
        upload();
    }
    void upload() { qg.upload(q); kg.upload(k); vg.upload(v); gg.upload(go); lengths.upload(valid); }
    void forward() {
        require(ar::forward(s,qg.get(),kg.get(),vg.get(),lengths.get(),out.get(),mx.get(),inv.get(),status.get()),"RDNA forward did not enqueue");
    }
    void backward() {
        require(ar::backward(s,qg.get(),kg.get(),vg.get(),out.get(),gg.get(),lengths.get(),
            mx.get(),inv.get(),status.get(),delta.get(),dq.get(),dk.get(),dv.get()),"RDNA VJP did not enqueue");
    }
    void padded_zeros() {
        const auto o = out.read(), gq = dq.read(), gk = dk.read(), gv = dv.read();
        const auto m = mx.read(), l = inv.read(), dt = delta.read();
        for (int b = 0; b < s.batch; ++b) for (int i = valid[b]; i < s.sequence; ++i) {
            for (int h = 0; h < s.query_heads; ++h) {
                require(m[row(s,b,i,h)] == 0 && l[row(s,b,i,h)] == 0 && dt[row(s,b,i,h)] == 0,"padded tape was not zero");
                for (int d = 0; d < s.head_dim; ++d) {
                    const auto x = index(s,b,i,h,s.query_heads,d);
                    require(o[x] == 0 && gq[x] == 0,"padded query output/gradient was not zero");
                }
            }
            for (int h = 0; h < s.kv_heads; ++h) for (int d = 0; d < s.head_dim; ++d) {
                const auto x = index(s,b,i,h,s.kv_heads,d);
                require(gk[x] == 0 && gv[x] == 0,"padded KV gradient was not zero");
            }
        }
    }
};

void parity(ar::Precision p, int seq, int dim, int heads, int kv, int window, std::vector<int> valid) {
    ar::Shape s{static_cast<int>(valid.size()),seq,heads,kv,dim,window,1.0f/std::sqrt(static_cast<float>(dim)),p};
    require(ar::supported(s),"RDNA3 rocWMMA capability is required (no successful skip)");
    Fixture f(s,std::move(valid));
    const auto r = reference(s,f.valid,effective(f.q,p),effective(f.k,p),effective(f.v,p),f.go);
    f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA forward/backward reference");
    for (int status : f.status.read()) require(status == 0,"valid attention fixture rejected");
    close(f.out.read(),r.out,"forward"); close(f.mx.read(),r.max,"row maximum");
    close(f.inv.read(),r.inv,"inverse sum"); close(f.delta.read(),r.delta,"delta");
    close(f.dq.read(),r.dq,"dQ"); close(f.dk.read(),r.dk,"dK"); close(f.dv.read(),r.dv,"dV");
    f.padded_zeros();
    const auto o = f.out.read(), gq = f.dq.read(), gk = f.dk.read(), gv = f.dv.read();
    f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA deterministic repeat");
    require(o == f.out.read() && gq == f.dq.read() && gk == f.dk.read() && gv == f.dv.read(),"owner reduction changed across identical repeats");
    std::cout << "RDNA reference precision=" << static_cast<int>(p) << " S=" << seq << " D=" << dim
              << " H/KV=" << heads << '/' << kv << " window=" << window << std::endl;
}

double loss(const Reference& r, const std::vector<float>& go) {
    double y = 0;
    for (std::size_t i = 0; i < go.size(); ++i) if (std::isfinite(go[i])) y += r.out[i]*go[i];
    return y;
}
void rounding_contract(ar::Precision p) {
    ar::Shape s{1,3,2,1,7,1,0.5f,p};
    Fixture f(s,{3});
    const float midpoint = p == ar::Precision::BF16 ? 1.0f/256 : 1.0f/2048;
    const std::vector<float> values{1+midpoint,1+3*midpoint,-1-midpoint,-1-3*midpoint,
        0.0f,std::ldexp(1.0f,-24),std::ldexp(1.0f,-25)};
    for (std::size_t x = 0; x < f.v.size(); ++x) f.v[x] = values[x%values.size()];
    f.vg.upload(f.v); f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA lowp ties/subnormal rounding");
    const auto out = f.out.read(), dq = f.dq.read(), dk = f.dk.read();
    for (int i = 0; i < s.sequence; ++i) for (int h = 0; h < s.query_heads; ++h)
        for (int d = 0; d < s.head_dim; ++d) {
            const auto qi = index(s,0,i,h,s.query_heads,d), vi = index(s,0,i,0,s.kv_heads,d);
            require(out[qi] == static_cast<float>(rounded(f.v[vi],p)),"device cast disagrees with independent RNE conversion");
            require(dq[qi] == 0,"one-key softmax has a spurious dQ");
        }
    for (float g : dk) require(g == 0,"one-key softmax has a spurious dK");
}
void derivatives(ar::Precision p) {
    ar::Shape s{1,5,4,2,7,4,0.41f,p};
    Fixture f(s,{5});
    auto q = effective(f.q,p), k = effective(f.k,p), v = effective(f.v,p);
    const auto r = reference(s,f.valid,q,k,v,f.go);
    f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA frozen lowp derivative");
    const auto dq = f.dq.read(), dk = f.dk.read(), dv = f.dv.read();
    const std::size_t qi = index(s,0,4,3,s.query_heads,2), ki = index(s,0,2,1,s.kv_heads,3);
    for (int kind = 0; kind < 3; ++kind) {
        auto& operand = kind == 0 ? q : kind == 1 ? k : v;
        const auto i = kind == 0 ? qi : ki;
        const auto& analytic = kind == 0 ? r.dq : kind == 1 ? r.dk : r.dv;
        const auto& actual = kind == 0 ? dq : kind == 1 ? dk : dv;
        const double old = operand[i], eps = 1e-5;
        // Freeze the effective cast, then differentiate the smooth operator.
        operand[i] = old+eps; const auto plus = reference(s,f.valid,q,k,v,f.go);
        operand[i] = old-eps; const auto minus = reference(s,f.valid,q,k,v,f.go);
        operand[i] = old;
        const double fd = (loss(plus,f.go)-loss(minus,f.go))/(2*eps);
        require(std::abs(fd-analytic[i]) < 1e-7,"independent frozen-lowp oracle finite difference failed");
        require(std::abs(fd-actual[i]) < 3e-5,"device frozen-lowp VJP disagrees with finite difference");
    }
    // Literal lowp casts are locally constant. VJP is explicitly STE, not a
    // misleading finite difference of the rounded forward itself.
    const float center = 0.3125f, eps = p == ar::Precision::BF16 ? 1e-5f : 1e-6f;
    require(rounded(center+eps,p) == rounded(center-eps,p),"quantizer plateau fixture invalid");
    require(std::abs(r.dv[ki]) > 1e-5,"STE gradient must differ from literal quantizer derivative");
}

void masks_and_failure(ar::Precision p) {
    ar::Shape s{2,19,4,2,17,7,0.25f,p};
    Fixture f(s,{19,11});
    // Invalid metadata is flagged per batch and never dereferenced as a length.
    f.valid[1] = s.sequence+1; f.lengths.upload(f.valid);
    f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA invalid prefix");
    require(f.status.read() == std::vector<int>({0,1}),"invalid prefix status changed");
    auto zero_batch = [](const std::vector<float>& x, std::size_t first) {
        for (std::size_t i = first; i < x.size(); ++i) require(x[i] == 0,"failed batch leaked output or gradient");
    };
    zero_batch(f.out.read(),f.q.size()/2); zero_batch(f.dq.read(),f.q.size()/2);
    zero_batch(f.dk.read(),f.k.size()/2); zero_batch(f.dv.read(),f.v.size()/2);
    zero_batch(f.mx.read(),ar::row_elements(s)/2); zero_batch(f.inv.read(),ar::row_elements(s)/2);
    zero_batch(f.delta.read(),ar::row_elements(s)/2);
    f.valid[1] = -1; f.lengths.upload(f.valid); f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA negative prefix");
    require(f.status.read() == std::vector<int>({0,1}),"negative prefix accepted");

    f.valid[1] = 11; f.lengths.upload(f.valid);
    const auto offset = index(s,1,0,0,s.query_heads);
    const float original = f.q[offset];
    for (float invalid : {65.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        f.q[offset] = invalid; f.qg.upload(f.q); f.forward(); f.backward();
        gpu_parity_test::cuda_sync_or_throw("RDNA invalid operand");
        require(f.status.read() == std::vector<int>({0,2}),"invalid operand status changed");
        zero_batch(f.out.read(),f.q.size()/2); zero_batch(f.dq.read(),f.q.size()/2);
        zero_batch(f.dk.read(),f.k.size()/2); zero_batch(f.dv.read(),f.v.size()/2);
    }
    f.q[offset] = original; f.qg.upload(f.q); f.forward();
    f.go[offset] = std::numeric_limits<float>::quiet_NaN(); f.gg.upload(f.go); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA invalid dO");
    require(f.status.read() == std::vector<int>({0,2}),"nonfinite dO was not rejected");
    zero_batch(f.dq.read(),f.q.size()/2); zero_batch(f.dk.read(),f.k.size()/2); zero_batch(f.dv.read(),f.v.size()/2);

    // K and V have their own preflight scans, independent of the query scan.
    const auto ko = index(s,1,0,0,s.kv_heads);
    for (int which = 0; which < 2; ++which) {
        auto& host = which == 0 ? f.k : f.v;
        auto& device = which == 0 ? f.kg : f.vg;
        const float saved = host[ko];
        host[ko] = std::numeric_limits<float>::quiet_NaN(); device.upload(host);
        f.forward(); f.backward();
        gpu_parity_test::cuda_sync_or_throw("RDNA invalid KV operand");
        require(f.status.read() == std::vector<int>({0,2}),"nonfinite K/V was not rejected");
        zero_batch(f.out.read(),f.q.size()/2); zero_batch(f.dk.read(),f.k.size()/2); zero_batch(f.dv.read(),f.v.size()/2);
        host[ko] = saved; device.upload(host);
    }

    require(!ar::forward(s,f.qg.get(),f.kg.get(),f.vg.get(),f.lengths.get(),f.qg.get(),f.mx.get(),f.inv.get(),f.status.get()),"aliased forward output accepted");
    require(!ar::backward(s,f.qg.get(),f.kg.get(),f.vg.get(),f.out.get(),f.gg.get(),f.lengths.get(),
        f.mx.get(),f.inv.get(),f.status.get(),f.delta.get(),f.dq.get(),f.dq.get(),f.dv.get()),"aliased gradients accepted");

    // Future keys and keys outside the window cannot affect an earlier row.
    Fixture m(s,{19,11});
    m.forward(); const auto before = m.out.read();
    for (int h = 0; h < s.kv_heads; ++h) for (int d = 0; d < s.head_dim; ++d) {
        m.k[index(s,0,0,h,s.kv_heads,d)] = 61; m.v[index(s,0,0,h,s.kv_heads,d)] = -61;
        m.k[index(s,0,18,h,s.kv_heads,d)] = -63; m.v[index(s,0,18,h,s.kv_heads,d)] = 63;
    }
    m.kg.upload(m.k); m.vg.upload(m.v); m.forward(); const auto after = m.out.read();
    for (int h = 0; h < s.query_heads; ++h) for (int d = 0; d < s.head_dim; ++d)
        require(before[index(s,0,10,h,s.query_heads,d)] == after[index(s,0,10,h,s.query_heads,d)],"window/causal masking leaked hidden keys");
}
void stability(ar::Precision p) {
    ar::Shape s{1,35,2,1,256,35,16.0f,p};
    Fixture f(s,{35});
    for (std::size_t x = 0; x < f.q.size(); ++x) {
        f.q[x] = (x%3 == 0 ? -64.0f : 64.0f);
        f.go[x] = 0.125f;
    }
    for (std::size_t x = 0; x < f.k.size(); ++x) {
        f.k[x] = (x%5 == 0 ? -64.0f : 64.0f);
        f.v[x] = static_cast<float>(static_cast<int>(x%9)-4)/16.0f;
    }
    f.upload();
    const auto r = reference(s,f.valid,effective(f.q,p),effective(f.k,p),effective(f.v,p),f.go);
    f.forward(); f.backward();
    gpu_parity_test::cuda_sync_or_throw("RDNA extreme finite online softmax");
    require(f.status.read() == std::vector<int>({0}),"bounded extreme fixture rejected");
    close(f.out.read(),r.out,"large-score forward",1e-4,1e-3);
    close(f.dq.read(),r.dq,"large-score dQ",2e-3,2e-3);
    close(f.dk.read(),r.dk,"large-score dK",2e-3,2e-3);
    close(f.dv.read(),r.dv,"large-score dV",1e-4,1e-3);
}
#endif

void host_geometry() {
    ar::Shape good{2,19,6,2,33,9,0.17f,ar::Precision::BF16};
    require(ar::geometry_eligible(good),"tail geometry rejected");
    require(ar::query_elements(good) == 2u*19*6*33 && ar::kv_elements(good) == 2u*19*2*33,"layout size helpers wrong");
    for (int kind = 0; kind < 10; ++kind) {
        auto bad = good;
        switch (kind) {
            case 0: bad.batch = 0; break;
            case 1: bad.sequence = 65535*16+1; break;
            case 2: bad.query_heads = 5; break;
            case 3: bad.kv_heads = 0; break;
            case 4: bad.head_dim = 257; break;
            case 5: bad.window = 0; break;
            case 6: bad.scale = std::numeric_limits<float>::quiet_NaN(); break;
            case 7: bad.precision = static_cast<ar::Precision>(0); break;
            case 8: bad.scale = 17; break;
            case 9: bad.batch = 65536; break;
        }
        require(!ar::geometry_eligible(bad) && !ar::supported(bad),"ineligible geometry accepted");
    }
    require(!ar::forward(good,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr),"null forward operands accepted");
}
} // namespace

int main() {
    return nsos::gpu_parity_test::run_parity("attention_rdna_training",[] {
        host_geometry();
#ifdef USE_CUDA
        for (auto p : {ar::Precision::BF16,ar::Precision::FP16}) {
            parity(p,1,1,2,1,1,{1,0});
            parity(p,15,15,2,2,1,{15,7,0});
            parity(p,16,16,4,1,64,{16,1});
            parity(p,17,17,6,2,5,{17,9,0});
            parity(p,17,32,4,2,2,{17,13});
            parity(p,33,33,4,2,19,{33,23,1});
            parity(p,17,64,2,1,8,{17,2});
            parity(p,35,65,3,1,100,{35,17});
            parity(p,17,128,2,1,17,{17,0});
            parity(p,19,129,4,2,3,{19,7,0});
            parity(p,19,256,4,2,18,{19,0});
            rounding_contract(p); derivatives(p); masks_and_failure(p); stability(p);
        }
#endif
    });
}
