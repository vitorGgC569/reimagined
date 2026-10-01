#include "gpu_parity_common.h"
#include "bitlinear.h"
#include "gpu_moe_training.h"
#include "cuda/moe_training_kernels.cuh"
#include "cuda/device_buffer.h"
#include "cuda/moe_training_wmma.cuh"
#include "gpu_execution.h"
#include "training_runtime_policy.h"
#include "../sparse_gradient_contract.h"
#include <memory>
#include <vector>
#include <iostream>
#include <limits>
#include <iomanip>

using namespace nsos;
using namespace nsos::gpu_parity_test;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void set_wmma(bool enabled) {
#ifdef _WIN32
    require(_putenv_s("NSOS_MOE_WMMA_TRAINING", enabled ? "1" : "0") == 0, "WMMA env fixture failed");
#else
    require(setenv("NSOS_MOE_WMMA_TRAINING", enabled ? "1" : "0", 1) == 0, "WMMA env fixture failed");
#endif
}
using Experts = std::vector<std::unique_ptr<BitLinear>>;
struct ScopedEnvironment {
    const char* key;
    std::string previous;
    bool present;
    explicit ScopedEnvironment(const char* name) : key(name) {
        const char* value = std::getenv(key);
        present = value != nullptr;
        if (value) previous = value;
    }
    void set(const char* value) {
#ifdef _WIN32
        require(_putenv_s(key, value) == 0, "cannot set sparse optimizer test policy");
#else
        require(setenv(key, value, 1) == 0, "cannot set sparse optimizer test policy");
#endif
    }
    ~ScopedEnvironment() {
#ifdef _WIN32
        _putenv_s(key, present ? previous.c_str() : "");
#else
        if (present) setenv(key, previous.c_str(), 1); else unsetenv(key);
#endif
    }
};
void sparse_optimizer_paths() {
    ScopedEnvironment grouped("NSOS_MOE_GROUPED_TRAINING"), ordered("NSOS_MOE_ORDERED_DEVICE"),
        fused("NSOS_FUSED_OPT"), chunks("NSOS_DETERMINISTIC_ADAMW_CHUNKED"),
        multi("NSOS_DETERMINISTIC_MULTI_TENSOR_OPT"), crit("NSOS_CRIT_REG"), crit_lr("NSOS_CRIT_LR");
    grouped.set("1"); ordered.set("1"); multi.set("1"); crit.set("0"); crit_lr.set("0");
    const bool previous_determinism = determinism::deterministic_reductions_enabled();
    const int previous_precision = matmul_precision_mode();
    set_matmul_precision_mode(training_policy::moe_wmma_training() ? 1 : 0);
    for (int path = 0; path < 4; ++path) {
        determinism::set_deterministic_reductions(path == 1 || path == 2);
        fused.set(path == 3 ? "0" : "1"); chunks.set(path == 2 ? "0" : "1");
        sparse_gradient_test::routed_optimizer_contract(Device::GPU);
    }
    determinism::set_deterministic_reductions(previous_determinism);
    set_matmul_precision_mode(previous_precision);
}
std::vector<BitLinear*> views(Experts& experts) {
    std::vector<BitLinear*> result;
    for (auto& e : experts) result.push_back(e.get());
    return result;
}
Experts make_experts(int in, int out, bool exact, bool qat, bool bias) {
    Experts result;
    for (int e = 0; e < 4; ++e) {
        auto op = std::make_unique<BitLinear>(in, out, bias, 100 + e);
        op->set_training_mode(true); op->set_reference_path(!qat);
        op->set_exact_linear_mode(exact); op->set_precision_mode(8);
        for (int i = 0; i < op->weight.data.size; ++i)
            op->weight.data.data()[i] = 0.025f * std::sin(0.131f * (i + 1 + e));
        for (int i = 0; i < out; ++i) {
            op->magnitude.data.data()[i] = 0.9f + 0.04f * std::cos(0.21f * i);
            if (bias) op->bias.data.data()[i] = 0.01f * std::sin(0.17f * (i + 1));
        }
        op->to(Device::GPU); result.push_back(std::move(op));
    }
    return result;
}

// Independent scheduling/oracle: existing per-expert BitLinear GPU paths,
// host fixture routing, host weighted combine. Never calls grouped helpers.
void case_parity(int mode, bool exact, bool qat, bool empty, bool bias, int R = 17, int D = 33, int H = 35) {
    constexpr int E = 4;
    set_matmul_precision_mode(mode);
    auto up = make_experts(D,H,exact,qat,bias), down = make_experts(H,D,exact,qat,bias);
    auto ref_up = make_experts(D,H,exact,qat,bias), ref_down = make_experts(H,D,exact,qat,bias);
    Tensor x({R,D},Device::CPU), grad(x.shape.dims,Device::CPU);
    Tensor routes = Tensor::zeros({R,E},Device::CPU);
    for (int i = 0; i < x.size; ++i) {
        x.data()[i] = 0.7f * std::sin(0.037f * (i + 1));
        grad.data()[i] = 0.08f * std::cos(0.043f * (i + 2));
    }
    if (!empty) for (int r = 0; r < R-1; ++r) {
        routes.data()[r*E] = 0.25f + 0.01f * (r%3);
        routes.data()[r*E+2] = 1 - routes.data()[r*E];
    }
    const Tensor xg = x.to(Device::GPU), rg = routes.to(Device::GPU), gg = grad.to(Device::GPU);
    Tensor expected = Tensor::zeros({R,D},Device::CPU), dx = expected.clone();
    Tensor gw = Tensor::zeros({R,E},Device::CPU);
    for (int e = 0; e < E; ++e) {
        std::vector<int> rows;
        for (int r = 0; r < R; ++r) if (routes.data()[r*E+e] != 0) rows.push_back(r);
        if (rows.empty()) continue;
        Tensor input({static_cast<int>(rows.size()),D},Device::CPU), dy(input.shape.dims,Device::CPU);
        for (size_t r = 0; r < rows.size(); ++r) for (int d = 0; d < D; ++d) {
            input.data()[r*D+d] = x.data()[rows[r]*D+d];
            dy.data()[r*D+d] = grad.data()[rows[r]*D+d] * routes.data()[rows[r]*E+e];
        }
        Tensor pre = ref_up[e]->forward(input.to(Device::GPU));
        Tensor out = ref_down[e]->forward(pre.squared_relu());
        Tensor dh = ref_down[e]->backward(dy.to(Device::GPU));
        Tensor dinput = ref_up[e]->backward(Tensor::squared_relu_backward(dh,pre)).cpu();
        Tensor host = out.cpu();
        for (size_t r = 0; r < rows.size(); ++r) {
            double dot = 0;
            for (int d = 0; d < D; ++d) {
                expected.data()[rows[r]*D+d] += host.data()[r*D+d] * routes.data()[rows[r]*E+e];
                dx.data()[rows[r]*D+d] += dinput.data()[r*D+d];
                dot += static_cast<double>(grad.data()[rows[r]*D+d]) * host.data()[r*D+d];
            }
            gw.data()[rows[r]*E+e] = static_cast<float>(dot);
        }
    }
    GpuMoeTraining grouped;
    const auto count_before = gpu::dispatch_counters()[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeWmmaGemm)];
    const auto commits_before = gpu::dispatch_counters()[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeGradientCommit)];
    reset_gpu_transfer_stats();
    Tensor out = grouped.forward(xg,rg,views(up),views(down),2);
    require(gpu_transfer_stats().d2h_calls == 0, "grouped forward downloaded routing/payload");
    reset_gpu_transfer_stats();
    auto [actual_dx, actual_gw] = grouped.backward(gg,true);
    const auto count_after = gpu::dispatch_counters()[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeWmmaGemm)];
    const auto commits_after = gpu::dispatch_counters()[static_cast<unsigned>(gpu::DispatchPath::GroupedMoeGradientCommit)];
    require(commits_after-commits_before==(empty?0u:2u),"grouped gradient commit bank count differs");
    const bool selected = training_policy::moe_wmma_training() && moe_training_wmma_geometry(R,D,H,mode);
    require(count_after - count_before == (selected ? 6u : 0u), "WMMA geometry dispatch count changed");
    const auto stats = gpu_transfer_stats();
    require(stats.d2h_calls == 1 && stats.d2h_bytes == (E+2)*sizeof(int), "grouped backward violated late metadata boundary");
    require(stats.stream_synchronizations == 1, "grouped backward added unexpected explicit stream fences");
    cuda_sync_or_throw("grouped MoE parity");
    // FP32 accumulation; low precision rounds the same operands as hipBLAS.
    const float atol = mode == 0 ? 2e-5f : 2e-4f;
    assert_close(expected,out,atol,"grouped forward",2e-3f);
    assert_close(dx,actual_dx,atol,"grouped input VJP",3e-3f);
    assert_close(gw,actual_gw,atol,"grouped router VJP",3e-3f);
    for (int e = 0; e < E; ++e) for (bool is_down : {false,true}) {
        auto a = (is_down ? down[e] : up[e])->parameters();
        auto b = (is_down ? ref_down[e] : ref_up[e])->parameters();
        require(a.size() == b.size(), "grouped parameter registry differs");
        for (size_t p = 0; p < a.size(); ++p) {
            require((a[p]->grad.size == 0) == (b[p]->grad.size == 0), "inactive expert gradient/Adam semantics changed");
            if (a[p]->grad.size) assert_close(b[p]->grad,a[p]->grad,atol,"grouped parameter VJP",3e-3f);
        }
    }
    bool consumed = false;
    try { grouped.backward(gg,true); } catch (const std::invalid_argument&) { consumed = true; }
    require(consumed,"grouped tape can be consumed twice");
    grouped.forward(xg,rg,views(up),views(down),2);
    ++up[0]->weight.version;
    bool stale = false;
    try { grouped.backward(gg,true); } catch (const std::runtime_error&) { stale = true; }
    require(stale,"grouped backward accepted a stale parameter version");
    grouped.discard_backward_state();
    std::cout << "grouped case mode=" << mode << " exact=" << exact << " qat=" << qat << " empty=" << empty << std::endl;
}

// Independent matrix oracle, not hipBLAS or another grouped implementation.
// Operands use exactly representable half/bfloat values; sums use double.
// Matrix-instruction reduction is not a serial IEEE FMA contract. Check the
// standard FP32 dot-product forward-error bound gamma_(2*paddedK)*sum(abs(a*b)),
// rather than claiming bitwise equality from exactly representable operands.
// Multiple CTAs, K tails, a seven-row segment and an empty NaN expert exercise
// LDS padding, segment bounds, all three products and no inactive writes.
void wmma_matrix_oracle(int mode) {
    constexpr int R = 56, D = 65, H = 67, E = 3;
    Tensor x({R,D},Device::CPU), dy({R,H},Device::CPU), w({E,H,D},Device::CPU);
    for (int i=0;i<x.size;++i) x.data()[i] = ((i*7)%31-15) / 32.0f;
    for (int i=0;i<dy.size;++i) dy.data()[i] = ((i*11)%23-11) / 32.0f;
    for (int i=0;i<w.size;++i) w.data()[i] = ((i*13)%29-14) / 32.0f;
    for (int i=H*D;i<2*H*D;++i) w.data()[i] = std::numeric_limits<float>::quiet_NaN();
    const Tensor xg=x.to(Device::GPU), dg=dy.to(Device::GPU), wg=w.to(Device::GPU);
    Tensor out=Tensor::ones({R,H},Device::GPU).mul(17), post=out.clone();
    Tensor dx=Tensor::ones({R,D},Device::GPU).mul(17);
    Tensor dw=Tensor::ones({E,H,D},Device::GPU).mul(17);
    std::vector<GpuMoeTrainingLinearView> host(E);
    for (int e=0;e<E;++e) { host[e].weight=wg.raw_data()+e*H*D; host[e].inputs=D; host[e].outputs=H; }
    cuda_detail::DeviceBuffer<GpuMoeTrainingLinearView> descriptors;
    cuda_detail::DeviceBuffer<int> offsets;
    const int positions[]={0,49,49,R,0};
    require(cudaMemcpy(descriptors.ensure(E),host.data(),E*sizeof(host[0]),cudaMemcpyHostToDevice)==cudaSuccess,"matrix descriptor upload failed");
    require(cudaMemcpy(offsets.ensure(E+2),positions,sizeof(positions),cudaMemcpyHostToDevice)==cudaSuccess,"matrix offsets upload failed");
    require(launch_moe_training_wmma_gemm(0,descriptors.get(),offsets.get(),xg.raw_data(),nullptr,out.raw_data(),post.raw_data(),nullptr,R,E,D,H,mode),"WMMA matrix forward failed");
    require(launch_moe_training_wmma_gemm(1,descriptors.get(),offsets.get(),dg.raw_data(),nullptr,dx.raw_data(),nullptr,nullptr,R,E,D,H,mode),"WMMA matrix dX failed");
    require(launch_moe_training_wmma_gemm(2,descriptors.get(),offsets.get(),dg.raw_data(),xg.raw_data(),dw.raw_data(),nullptr,nullptr,R,E,D,H,mode),"WMMA matrix dW failed");
    const Tensor actual=out.cpu(), actual_post=post.cpu(), actual_dx=dx.cpu(), actual_dw=dw.cpu();
    double worst_error=0, worst_fraction=0;
    const auto check_dot=[&](float observed, double expected, double absolute_sum, int k) {
        const int operations=2*((k+31)/32)*32;
        const double u=std::numeric_limits<float>::epsilon()/2.0;
        const double bound=(operations*u/(1-operations*u))*absolute_sum;
        const double error=std::abs(static_cast<double>(observed)-expected);
        worst_error=std::max(worst_error,error);
        worst_fraction=std::max(worst_fraction,bound?error/bound:0);
        if (!std::isfinite(observed) || error>bound) {
            std::cerr << std::setprecision(17) << "WMMA oracle mode=" << mode << " actual=" << observed
                      << " expected=" << expected << " fp32_bound=" << bound << std::endl;
            throw std::runtime_error("WMMA differs from independent double oracle beyond FP32 accumulation bound");
        }
    };
    for (int e=0;e<E;++e) {
        const int begin=positions[e], count=positions[e+1]-begin;
        for (int r=begin;r<begin+count;++r) {
            for (int h=0;h<H;++h) {
                double expected=0, absolute_sum=0;
                for (int d=0;d<D;++d) {
                    const double product=static_cast<double>(x.data()[r*D+d])*w.data()[(e*H+h)*D+d];
                    expected+=product; absolute_sum+=std::abs(product);
                }
                check_dot(actual.data()[r*H+h],expected,absolute_sum,D);
                require(actual_post.data()[r*H+h]==actual.data()[r*H+h],"WMMA exact epilogue changed");
            }
            for (int d=0;d<D;++d) {
                double expected=0, absolute_sum=0;
                for (int h=0;h<H;++h) {
                    const double product=static_cast<double>(dy.data()[r*H+h])*w.data()[(e*H+h)*D+d];
                    expected+=product; absolute_sum+=std::abs(product);
                }
                check_dot(actual_dx.data()[r*D+d],expected,absolute_sum,H);
            }
        }
        for (int h=0;h<H;++h) for (int d=0;d<D;++d) {
            double expected=0, absolute_sum=0;
            for (int r=begin;r<begin+count;++r) {
                const double product=static_cast<double>(dy.data()[r*H+h])*x.data()[r*D+d];
                expected+=product; absolute_sum+=std::abs(product);
            }
            if (count) check_dot(actual_dw.data()[(e*H+h)*D+d],expected,absolute_sum,count);
            else require(actual_dw.data()[(e*H+h)*D+d]==17,"WMMA wrote inactive NaN expert");
        }
    }
    std::cout << "WMMA double oracle mode=" << mode << " max_error=" << worst_error
              << " max_fraction_of_fp32_bound=" << worst_fraction << std::endl;
    require(!launch_moe_training_wmma_gemm(3,descriptors.get(),offsets.get(),xg.raw_data(),nullptr,out.raw_data(),post.raw_data(),nullptr,R,E,D,H,mode),"invalid WMMA kind accepted");
}

void compute_policy_tape_guard() {
    const bool enabled=training_policy::moe_wmma_training();
    auto up=make_experts(33,35,true,false,true), down=make_experts(35,33,true,false,true);
    Tensor x=Tensor::ones({17,33},Device::GPU), routes=Tensor::zeros({17,4},Device::CPU);
    for (int r=0;r<17;++r) routes.data()[r*4]=1;
    GpuMoeTraining grouped;
    grouped.forward(x,routes.to(Device::GPU),views(up),views(down),1);
    set_wmma(!enabled);
    bool rejected=false;
    try { grouped.backward(x,false); } catch (const std::runtime_error&) { rejected=true; }
    catch (...) { set_wmma(enabled); throw; }
    set_wmma(enabled);
    require(rejected,"grouped tape accepted changed WMMA policy");
    for (auto& e:up) require(e->weight.grad.size==0,"policy rejection committed gradients");
    rejected=false;
    try { grouped.backward(x,false); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"failed WMMA backward retained a reusable tape");
    grouped.forward(x,routes.to(Device::GPU),views(up),views(down),1);
    grouped.backward(x,false);
    grouped.forward(x,routes.to(Device::GPU),views(up),views(down),1);
    const int previous=matmul_precision_mode();
    set_matmul_precision_mode(previous==0?1:0);
    rejected=false;
    try { grouped.backward(x,false); } catch (const std::runtime_error&) { rejected=true; }
    catch (...) { set_matmul_precision_mode(previous); throw; }
    set_matmul_precision_mode(previous);
    require(rejected,"grouped tape accepted changed precision");
    rejected=false;
    try { grouped.backward(x,false); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"failed precision backward retained a reusable tape");
    grouped.forward(x,routes.to(Device::GPU),views(up),views(down),1);
    grouped.backward(x,false);
    grouped.discard_backward_state();
}
void invalid_capacity() {
    auto up = make_experts(17,19,true,false,true), down = make_experts(19,17,true,false,true);
    Tensor x = Tensor::ones({3,17},Device::GPU), routes = Tensor::ones({3,4},Device::GPU);
    GpuMoeTraining grouped;
    Tensor output = grouped.forward(x,routes,views(up),views(down),1).cpu();
    for (int i = 0; i < output.size; ++i) require(std::isnan(output.data()[i]),"invalid capacity produced plausible output");
    bool rejected = false;
    try { grouped.backward(x,true); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected,"invalid sparse capacity was accepted");
    for (auto& e : up) require(e->weight.grad.size == 0,"invalid routing committed gradients");
    cuda_sync_or_throw("invalid capacity guard");
}

void active_qat_preparation() {
    constexpr int E = 3, N = 1041, Blocks = (N + 255) / 256;
    Tensor latent({E, N}, Device::CPU);
    for (int i = 0; i < latent.size; ++i) latent.data()[i] = 0.017f * std::sin(0.09f * (i + 1));
    for (int i = 2 * N; i < 3 * N; ++i) latent.data()[i] = std::numeric_limits<float>::quiet_NaN();
    Tensor weights = Tensor::ones({E, N}, Device::GPU).mul(17);
    Tensor scales = Tensor::ones({E}, Device::GPU).mul(19);
    const Tensor source = latent.to(Device::GPU);
    std::vector<GpuMoeTrainingLinearView> host(E);
    for (int e = 0; e < E; ++e) {
        host[e].weight = weights.raw_data() + e * N;
        host[e].latent_weight = source.raw_data() + e * N;
        host[e].qat_scale = e == 1 ? nullptr : scales.raw_data() + e;
        host[e].inputs = N; host[e].outputs = 1;
    }
    cuda_detail::DeviceBuffer<GpuMoeTrainingLinearView> descriptors;
    cuda_detail::DeviceBuffer<int> offsets;
    cuda_detail::DeviceBuffer<float> partials;
    const int positions[] = {0, 1, 2, 2, 0};
    require(cudaMemcpy(descriptors.ensure(E), host.data(), E * sizeof(host[0]), cudaMemcpyHostToDevice) == cudaSuccess,
            "QAT descriptor fixture upload failed");
    require(cudaMemcpy(offsets.ensure(E + 2), positions, sizeof(positions), cudaMemcpyHostToDevice) == cudaSuccess,
            "QAT offset fixture upload failed");
    require(launch_moe_training_prepare_qat(descriptors.get(), offsets.get(), partials.ensure(E * Blocks), E, N),
            "Active QAT preparation launch failed");
    cuda_sync_or_throw("active QAT preparation");
    const Tensor quantized = weights.cpu(), actual_scales = scales.cpu();
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += std::abs(latent.data()[i]);
    const float scale = static_cast<float>(sum / N) + 1e-8f;
    require(std::abs(actual_scales.data()[0] - scale) < 1e-7f, "Active QAT scale differs from independent absmean");
    for (int i = 0; i < N; ++i) {
        const float value = latent.data()[i] / (scale + 1e-8f);
        const float expected = (value > 0.5f ? 1 : value < -0.5f ? -1 : 0) * scale;
        require(std::abs(quantized.data()[i] - expected) < 1e-7f, "Active QAT ternary rule changed");
        require(quantized.data()[N + i] == 17 && quantized.data()[2 * N + i] == 17,
                "QAT wrote reference/inactive expert workspace");
    }
    require(actual_scales.data()[1] == 19 && actual_scales.data()[2] == 19,
            "QAT consumed inactive or reference scale");
    const Tensor before = weights.cpu();
    require(launch_moe_training_prepare_qat(descriptors.get(), offsets.get(), partials.get(), E, N), "QAT repeat failed");
    const Tensor after = weights.cpu();
    for (int i = 0; i < before.size; ++i) require(before.data()[i] == after.data()[i], "QAT reduction is nondeterministic");
    // Switching reference -> deferred QAT must allocate a distinct destination.
    auto expert = make_experts(17, 19, true, false, true);
    GpuMoeTrainingLinearView view;
    Tensor effective, qat_scale;
    expert[0]->prepare_gpu_grouped_training_view(view, effective, qat_scale, true);
    require(view.weight == view.latent_weight, "Reference view should borrow master weight");
    expert[0]->set_reference_path(false);
    expert[0]->prepare_gpu_grouped_training_view(view, effective, qat_scale, true);
    require(view.weight != view.latent_weight && view.qat_scale, "Deferred QAT destination aliases master weight");
}

void gradient_accumulation_oracle() {
    constexpr int E = 3, N = 1041, O = 17;
    Tensor w({E,N},Device::CPU), b({E,O},Device::CPU), m({E,O},Device::CPU);
    for (int i=0;i<w.size;++i) w.data()[i] = (i%31-15)/32.0f;
    for (int i=0;i<b.size;++i) { b.data()[i]=(i%17-8)/16.0f; m.data()[i]=(i%13-6)/8.0f; }
    for (int i=N;i<2*N;++i) w.data()[i]=std::numeric_limits<float>::quiet_NaN();
    for (int i=O;i<2*O;++i) b.data()[i]=m.data()[i]=std::numeric_limits<float>::quiet_NaN();
    const Tensor wg=w.to(Device::GPU), bg=b.to(Device::GPU), mg=m.to(Device::GPU);
    Tensor dw=Tensor::ones({E,N},Device::GPU).mul(17), db=Tensor::ones({E,O},Device::GPU).mul(19),
        dm=Tensor::ones({E,O},Device::GPU).mul(23);
    std::vector<GpuMoeGradientView> host(E);
    for (int e=0;e<E;++e) {
        host[e].weight=dw.raw_data()+e*N;
        host[e].bias=e==2?nullptr:db.raw_data()+e*O;
        host[e].magnitude=dm.raw_data()+e*O;
        host[e].add_mask=e==0?5u:4u;
    }
    cuda_detail::DeviceBuffer<GpuMoeGradientView> descriptors;
    cuda_detail::DeviceBuffer<int> offsets;
    int positions[]={0,2,2,3,0};
    require(cudaMemcpy(descriptors.ensure(E),host.data(),E*sizeof(host[0]),cudaMemcpyHostToDevice)==cudaSuccess,
            "gradient oracle descriptor upload failed");
    require(cudaMemcpy(offsets.ensure(E+2),positions,sizeof(positions),cudaMemcpyHostToDevice)==cudaSuccess,
            "gradient oracle offsets upload failed");
    require(launch_moe_training_accumulate_gradients(descriptors.get(),offsets.get(),wg.raw_data(),bg.raw_data(),
        mg.raw_data(),E,N,O),"gradient accumulation kernel failed");
    const Tensor aw=dw.cpu(), ab=db.cpu(), am=dm.cpu();
    for (int e=0;e<E;++e) {
        for (int i=0;i<N;++i) require(aw.data()[e*N+i]==(e==1?17:e==0?17+w.data()[e*N+i]:w.data()[e*N+i]),
                "gradient weight first/add/inactive oracle differs");
        for (int i=0;i<O;++i) {
            require(ab.data()[e*O+i]==(e==0?b.data()[i]:19),"optional/inactive bias was fetched or overwritten");
            require(am.data()[e*O+i]==(e==1?23:23+m.data()[e*O+i]),"gradient magnitude oracle differs");
        }
    }
    // Routing error must suppress all writes, including active destinations.
    positions[E+1]=1;
    require(cudaMemcpy(offsets.get(),positions,sizeof(positions),cudaMemcpyHostToDevice)==cudaSuccess,
            "gradient oracle invalid offsets upload failed");
    require(launch_moe_training_accumulate_gradients(descriptors.get(),offsets.get(),wg.raw_data(),bg.raw_data(),
        mg.raw_data(),E,N,O),"guarded gradient repeat failed");
    sparse_gradient_test::same(dw,aw,"invalid routing wrote weight gradients");
    sparse_gradient_test::same(db,ab,"invalid routing wrote bias gradients");
    sparse_gradient_test::same(dm,am,"invalid routing wrote magnitude gradients");
    require(!launch_moe_training_accumulate_gradients(descriptors.get(),offsets.get(),wg.raw_data(),bg.raw_data(),
        mg.raw_data(),1025,N,O),"invalid gradient expert geometry accepted");
}

void grouped_accumulation_contract() {
    constexpr int R=17,D=33,H=35,E=4;
    auto up=make_experts(D,H,false,true,true), down=make_experts(H,D,false,true,true);
    auto ref_up=make_experts(D,H,false,true,true), ref_down=make_experts(H,D,false,true,true);
    Tensor x=Tensor::ones({R,D},Device::GPU), grad=Tensor::ones({R,D},Device::GPU).mul(0.03f);
    Tensor r1=Tensor::zeros({R,E},Device::CPU), r2=r1.clone();
    for (int r=0;r<R;++r) {
        r1.data()[r*E]=0.25f; r1.data()[r*E+2]=0.75f;
        r2.data()[r*E]=0.5f; r2.data()[r*E+1]=0.5f;
    }
    const Tensor first_route=r1.to(Device::GPU), second_route=r2.to(Device::GPU);
    GpuMoeTraining grouped, control;
    grouped.forward(x,first_route,views(up),views(down),2); grouped.backward(grad,true);
    std::vector<Tensor> first;
    std::vector<const float*> addresses;
    for (bool is_down:{false,true}) for (int e=0;e<E;++e)
        for (auto* p:(is_down?down[e]:up[e])->parameters()) {
            first.push_back(p->grad.size?p->grad.cpu():Tensor()); addresses.push_back(p->grad.raw_data());
        }
    control.forward(x,second_route,views(ref_up),views(ref_down),2); control.backward(grad,true);
    reset_gpu_transfer_stats();
    grouped.forward(x,second_route,views(up),views(down),2); grouped.backward(grad,true);
    require(gpu_transfer_stats().d2d_calls==1,"grouped accumulation retained per-parameter D2D copies");
    size_t k=0;
    for (bool is_down:{false,true}) for (int e=0;e<E;++e) {
        auto actual=(is_down?down[e]:up[e])->parameters();
        auto reference=(is_down?ref_down[e]:ref_up[e])->parameters();
        for (size_t p=0;p<actual.size();++p,++k) {
            Tensor expected=first[k];
            if (reference[p]->has_gradient())
                expected=expected.size?expected.add(reference[p]->grad.cpu()):reference[p]->grad.cpu();
            require(actual[p]->has_gradient()==(expected.size!=0),"microbatch union activity differs");
            if (expected.size) sparse_gradient_test::same(actual[p]->grad,expected,"microbatch accumulation differs bitwise");
            if (first[k].size) require(actual[p]->grad.raw_data()==addresses[k],"grouped accumulation replaced stable gradient buffer");
        }
    }
    // Clear contribution flags but poison retained storage: the next first
    // contribution must overwrite, not depend on redundant physical zeros.
    for (bool is_down:{false,true}) for (int e=0;e<E;++e)
        for (auto* p:(is_down?down[e]:up[e])->parameters()) {
            p->zero_grad();
            if (p->grad.size) p->grad.copy_from(Tensor::ones(p->grad.shape.dims,Device::GPU).mul(17));
        }
    grouped.forward(x,second_route,views(up),views(down),2); grouped.backward(grad,true);
    for (bool is_down:{false,true}) for (int e=0;e<E;++e) {
        auto actual=(is_down?down[e]:up[e])->parameters();
        auto reference=(is_down?ref_down[e]:ref_up[e])->parameters();
        for (size_t p=0;p<actual.size();++p) {
            require(actual[p]->has_gradient()==reference[p]->has_gradient(),"reactivated activity differs");
            if (reference[p]->has_gradient()) sparse_gradient_test::same(actual[p]->grad,reference[p]->grad,
                "first grouped contribution read stale storage");
            else if (actual[p]->grad.size) sparse_gradient_test::same(actual[p]->grad,
                Tensor::ones(actual[p]->grad.shape.dims,Device::GPU).mul(17),"inactive expert destination overwritten");
        }
    }
    grouped.forward(x,second_route,views(up),views(down),2); grouped.backward(grad,true);
    reset_gpu_transfer_stats();
    grouped.forward(x,second_route,views(up),views(down),2); grouped.backward(grad,true);
    require(gpu_transfer_stats().h2d_calls==0,"stable grouped gradient descriptors were uploaded again");
}

void gradient_commit_preflight() {
    auto up=make_experts(33,35,false,true,true), down=make_experts(35,33,false,true,true);
    Tensor x=Tensor::ones({17,33},Device::GPU), route=Tensor::zeros({17,4},Device::CPU);
    for (int r=0;r<17;++r) route.data()[r*4]=1;
    const Tensor routing=route.to(Device::GPU);
    up[0]->track_gradient_contributions(); down[0]->track_gradient_contributions();
    up[0]->weight.add_grad(Tensor::ones({35,33},Device::GPU).mul(17));
    const Tensor before=up[0]->weight.grad.clone();
    down[0]->bias.grad=Tensor::ones({1},Device::GPU);
    GpuMoeTraining grouped;
    grouped.forward(x,routing,views(up),views(down),1);
    bool rejected=false;
    try { grouped.backward(x,true); } catch (const std::logic_error&) { rejected=true; }
    require(rejected,"invalid down-bank gradient destination accepted");
    sparse_gradient_test::same(up[0]->weight.grad,before,"preflight failure changed up-bank contribution");
    require(up[0]->weight.has_gradient() && !down[0]->weight.has_gradient(),
            "preflight failure published a partial contribution");
    down[0]->bias.grad=Tensor();
    // Alias is shape-valid but cannot be updated in parallel across banks.
    down[0]->weight.grad=up[0]->weight.grad.reshape({33,35});
    grouped.forward(x,routing,views(up),views(down),1);
    rejected=false;
    try { grouped.backward(x,true); } catch (const std::logic_error&) { rejected=true; }
    require(rejected,"aliased expert gradient destinations accepted");
    sparse_gradient_test::same(up[0]->weight.grad,before,"alias rejection changed contribution");
    require(!down[0]->weight.has_gradient(),"alias rejection published down-bank activity");
}
}
int main() {
    return run_parity("grouped_moe_training", [] {
        set_strict_gpu_execution(true);
        sparse_gradient_test::parameter_contract(Device::GPU);
        sparse_optimizer_paths();
        for (int mode : {0,1,2}) for (bool exact : {false,true}) for (bool qat : {false,true})
            case_parity(mode,exact,qat,false,true);
        case_parity(0,true,false,true,false);
        case_parity(0,false,true,true,true);
        invalid_capacity();
        active_qat_preparation();
        gradient_accumulation_oracle();
        grouped_accumulation_contract();
        gradient_commit_preflight();
        compute_policy_tape_guard();
        if (training_policy::moe_wmma_training()) {
            require(moe_training_wmma_supported(),"requested WMMA unavailable");
            for (int mode:{1,2}) {
                wmma_matrix_oracle(mode);
                case_parity(mode,true,false,false,true,50,65,67);
                case_parity(mode,false,true,false,true,50,65,67);
                case_parity(mode,true,false,false,false,7,33,35);
                case_parity(mode,true,false,true,false);
            }
        }
        set_matmul_precision_mode(0);
    });
}
