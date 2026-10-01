#include "gpu_parity_common.h"
#include "cuda/device_buffer.h"
#include "cuda/kernels.cuh"
#include "mamba2.h"
#include "jamba.h"
#include "nsos/determinism.h"
#include "training_runtime_policy.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace nsos;
using namespace nsos::gpu_parity_test;

namespace {
void require(bool ok, const char* label) {
    if (!ok) throw std::runtime_error(label);
}
void set_env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value) == 0, "environment mutation failed");
#else
    require(setenv(key, value, 1) == 0, "environment mutation failed");
#endif
}
template<class T> void upload(T* dst, const T* src, size_t count) {
    require(dst && cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess,
            "test metadata upload failed");
}
void fill(Tensor& x, float scale, float frequency) {
    for (int i = 0; i < x.size; ++i) x.data()[i] = scale * std::sin(frequency * (i + 1));
}
size_t index(int b, int s, int h, int S, int H, int hd) {
    return ((static_cast<size_t>(b)*S+s)*H+h)*hd;
}

void attention_case(int S, int H, int KV, int hd, int window) {
    constexpr int B = 3;
    const int group = H / KV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const int valid[B] = {S, std::max(1, S-5), 0};
    Tensor q({B,S,H,hd}, Device::CPU), k({B,S,KV,hd}, Device::CPU), v(k.shape.dims, Device::CPU);
    Tensor grad(q.shape.dims, Device::CPU);
    fill(q, 0.7f, 0.017f); fill(k, 0.6f, 0.031f); fill(v, 0.8f, 0.043f); fill(grad, 0.3f, 0.029f);
    Tensor out = Tensor::zeros(q.shape.dims, Device::CPU);
    Tensor dq = Tensor::zeros(q.shape.dims, Device::CPU);
    Tensor dk = Tensor::zeros(k.shape.dims, Device::CPU), dv = Tensor::zeros(k.shape.dims, Device::CPU);
    std::vector<double> probability(S), dp(S);
    for (int b=0; b<B; ++b) for (int i=0; i<valid[b]; ++i) for (int h=0; h<H; ++h) {
        const int kh = std::min(h/group, KV-1), first = std::max(0, i-window+1);
        const size_t qi = index(b,i,h,S,H,hd);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int j=first; j<=i; ++j) {
            double score=0, dot=0;
            const size_t ki=index(b,j,kh,S,KV,hd);
            for (int d=0; d<hd; ++d) {
                score += static_cast<double>(q.data()[qi+d])*k.data()[ki+d];
                dot += static_cast<double>(grad.data()[qi+d])*v.data()[ki+d];
            }
            probability[j]=score*scale; dp[j]=dot; maximum=std::max(maximum,probability[j]);
        }
        double denominator=0;
        for (int j=first; j<=i; ++j) { probability[j]=std::exp(probability[j]-maximum); denominator+=probability[j]; }
        double delta=0;
        for (int j=first; j<=i; ++j) { probability[j]/=denominator; delta+=probability[j]*dp[j]; }
        for (int d=0; d<hd; ++d) {
            double y=0, dx=0;
            for (int j=first; j<=i; ++j) {
                const size_t ki=index(b,j,kh,S,KV,hd)+d;
                const double ds=probability[j]*(dp[j]-delta)*scale;
                y+=probability[j]*v.data()[ki]; dx+=ds*k.data()[ki];
                dk.data()[ki]+=static_cast<float>(ds*q.data()[qi+d]);
                dv.data()[ki]+=static_cast<float>(probability[j]*grad.data()[qi+d]);
            }
            out.data()[qi+d]=static_cast<float>(y); dq.data()[qi+d]=static_cast<float>(dx);
        }
    }
    const Tensor qg=q.to(Device::GPU), kg=k.to(Device::GPU), vg=v.to(Device::GPU), gg=grad.to(Device::GPU);
    Tensor yg=Tensor::uninitialized(q.shape.dims,Device::GPU), lse=Tensor::uninitialized({B,S,H},Device::GPU);
    Tensor dxg=Tensor::uninitialized(q.shape.dims,Device::GPU), dkg=Tensor::uninitialized(k.shape.dims,Device::GPU);
    Tensor dvg=Tensor::uninitialized(k.shape.dims,Device::GPU), delta=Tensor::uninitialized({B,S,H},Device::GPU);
    cuda_detail::DeviceBuffer<int> lengths;
    int* dl=lengths.ensure(B); upload(dl,valid,B);
    require(launch_attn_tiled_forward(qg.raw_data(),kg.raw_data(),vg.raw_data(),dl,yg.raw_data(),lse.raw_data(),
        B,S,H,KV,hd,group,window,scale), "tiled attention forward rejected valid inputs");
    require(launch_attn_tiled_backward(qg.raw_data(),kg.raw_data(),vg.raw_data(),yg.raw_data(),gg.raw_data(),
        lse.raw_data(),dl,delta.raw_data(),dxg.raw_data(),dkg.raw_data(),dvg.raw_data(),
        B,S,H,KV,hd,group,window,scale), "tiled attention backward rejected valid inputs");
    cuda_sync_or_throw("tiled attention forward/backward");
    assert_close(out,yg,2e-5f,"tiled attention output",2e-5f);
    assert_close(dq,dxg,2e-5f,"tiled attention dQ",2e-5f);
    assert_close(dk,dkg,3e-5f,"tiled attention dK",3e-5f);
    assert_close(dv,dvg,3e-5f,"tiled attention dV",3e-5f);
    require(!launch_attn_tiled_forward(qg.raw_data(),kg.raw_data(),vg.raw_data(),dl,yg.raw_data(),lse.raw_data(),
        B,S,H,KV,hd,group,window,std::numeric_limits<float>::quiet_NaN()), "attention accepted a NaN scale");
}

void norm_case() {
    Tensor a({8205},Device::CPU), b({7},Device::CPU);
    fill(a,0.7f,0.023f); fill(b,0.6f,0.15f);
    double reference=0;
    for (const Tensor* x : {&a,&b}) for (int i=0;i<x->size;++i) reference+=static_cast<double>(x->data()[i])*x->data()[i];
    Tensor ag=a.to(Device::GPU), bg=b.to(Device::GPU);
    float* pointers[2]={ag.raw_data(),bg.raw_data()};
    const NsosMultiTensorChunk chunks[3]={{0,8192,0},{8192,13,0},{0,7,1}};
    cuda_detail::DeviceBuffer<float*> pg;
    cuda_detail::DeviceBuffer<NsosMultiTensorChunk> cg;
    cuda_detail::DeviceBuffer<double> total, partials;
    cuda_detail::DeviceBuffer<float> coefficient;
    cuda_detail::DeviceBuffer<int> issue;
    upload(pg.ensure(2),pointers,2); upload(cg.ensure(3),chunks,3);
    int ok=0; upload(issue.ensure(1),&ok,1);
    require(launch_chunked_norm_device_clip(total.ensure(1),partials.ensure(3),coefficient.ensure(1),
        pg.ensure(2),cg.ensure(3),3,1.0f,issue.ensure(1)),"chunked norm launch failed");
    cuda_sync_or_throw("chunked norm and clip");
    double observed=0;
    require(cudaMemcpy(&observed,total.ensure(1),sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess,"norm download failed");
    require(std::abs(observed-reference)<1e-9*reference,"chunked FP64 norm mismatch");
    const float scale=1.0f/(static_cast<float>(std::sqrt(reference))+1e-6f);
    assert_close(a.mul(scale),ag,1e-7f,"device clipping first tensor",1e-6f);
    assert_close(b.mul(scale),bg,1e-7f,"device clipping tail tensor",1e-6f);
    ag.copy_from(a.to(Device::GPU)); bg.copy_from(b.to(Device::GPU));
    int poisoned=1; upload(issue.ensure(1),&poisoned,1);
    require(launch_chunked_norm_device_clip(total.ensure(1),partials.ensure(3),coefficient.ensure(1),
        pg.ensure(2),cg.ensure(3),3,1.0f,issue.ensure(1)),"finite-issue norm launch failed");
    cuda_sync_or_throw("poisoned clipping");
    require(cudaMemcpy(&observed,total.ensure(1),sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess,"status download failed");
    require(observed<0,"deferred finite issue was lost");
    assert_close(a,ag,0,"finite issue must not clip gradients");
    assert_close(b,bg,0,"finite issue must not clip tail gradients");
}

void mamba_case(int S, bool checkpoint) {
    constexpr int D=64;
    MambaConfig cfg;
    cfg.faithful_mamba2=true; cfg.expand=2; cfg.head_dim=64;
    cfg.n_groups=1; cfg.conv_kernel=4; cfg.recompute_ssd=checkpoint;
    Mamba2SSD cpu(D,64,1,cfg), gpu(D,64,1,cfg);
    gpu.to(Device::GPU);
    auto cp=cpu.parameters(), gp=gpu.parameters();
    require(cp.size()==gp.size(),"Mamba parameter registry mismatch");
    for (size_t i=0;i<cp.size();++i) gp[i]->copy_data_from(cp[i]->data.to(Device::GPU));
    Tensor x({2,S,D},Device::CPU), g(x.shape.dims,Device::CPU);
    fill(x,0.11f,0.017f); fill(g,0.07f,0.013f);
    Context cc,gc;
    const Tensor expected=cpu.forward(x), actual=gpu.forward(x.to(Device::GPU));
    const size_t boundary_bytes=static_cast<size_t>(2)*((S+31)/32)*(D*2)*64*sizeof(float);
    require(gpu.faithful_peak_state_history_bytes()==boundary_bytes,"Mamba did not allocate compact boundary history");
    const Tensor dx=cpu.backward(g,cc), dxg=gpu.backward(g.to(Device::GPU),gc);
    cuda_sync_or_throw("boundary Mamba");
    require(gpu.faithful_peak_state_history_bytes()==boundary_bytes,"Mamba backward restored a dense history");
    assert_close(expected,actual,2e-3f,"boundary Mamba output",1e-3f);
    assert_close(dx,dxg,3e-3f,"boundary Mamba dInput",1e-3f);
    for (size_t i=0;i<cp.size();++i) {
        if (cp[i]->grad.size==0 && gp[i]->grad.size==0) continue;
        require(cp[i]->grad.size>0 && gp[i]->grad.size>0,"boundary Mamba missing parameter gradient");
        assert_close(cp[i]->grad,gp[i]->grad,3e-3f,cp[i]->name.c_str(),1e-3f);
    }
}

void moe_load_case() {
    set_env("NSOS_MOE_ORDERED_DEVICE","1");
    MoERouter cpu(16,4,2), gpu(16,4,2);
    gpu.to(Device::GPU);
    auto cp=cpu.parameters(), gp=gpu.parameters();
    require(cp.size()==gp.size(),"MoE load registry mismatch");
    for (size_t i=0;i<cp.size();++i) gp[i]->copy_data_from(cp[i]->data.to(Device::GPU));
    Tensor x({5,16},Device::CPU); fill(x,0.3f,0.027f);
    const Tensor xg=x.to(Device::GPU);
    const std::vector<uint8_t> mask={1,1,0,1,0};
    cpu.begin_aux_accumulation(); gpu.begin_aux_accumulation();
    for (int step=0;step<2;++step) {
        const auto a=cpu.forward(x,mask), b=gpu.forward(xg,mask);
        assert_close(a.second,b.second,1e-4f,"ordered MoE masked routing");
        gpu.materialize_expert_loads();
        for (size_t e=0;e<cpu.expert_loads.size();++e)
            require(std::abs(cpu.expert_loads[e]-gpu.expert_loads[e])<1e-4f,"MoE latest load telemetry mismatch");
    }
    cpu.finalize_aux_accumulation(); gpu.finalize_aux_accumulation();
    for (size_t e=0;e<cpu.expert_loads.size();++e)
        require(std::abs(cpu.expert_loads[e]-gpu.expert_loads[e])<2e-4f,"MoE accumulated load mismatch");
    gpu.begin_aux_accumulation(); (void)gpu.forward(xg,mask); gpu.cancel_aux_accumulation();
    gpu.begin_aux_accumulation(); (void)gpu.forward(xg,mask); gpu.finalize_aux_accumulation();
    for (size_t e=0;e<cpu.expert_loads.size();++e)
        require(std::abs(cpu.expert_loads[e]*0.5f-gpu.expert_loads[e])<1e-4f,"MoE cancellation leaked previous loads");
}

void moe_aux_device_case(bool ordered) {
    set_env("NSOS_MOE_ORDERED_DEVICE", ordered ? "1" : "0");
    set_matmul_precision_mode(0);
    MoERouter cpu(16, 5, 2), gpu(16, 5, 2);
    cpu.gate->set_reference_path(true); gpu.gate->set_reference_path(true);
    gpu.to(Device::GPU);
    const auto cp = cpu.parameters(), gp = gpu.parameters();
    for (size_t p = 0; p < cp.size(); ++p) gp[p]->copy_data_from(cp[p]->data.to(Device::GPU));
    cpu.begin_aux_accumulation(); gpu.begin_aux_accumulation();
    for (int rows : {3, 7}) {
        Tensor x({rows, 16}, Device::CPU); fill(x, 0.3f, 0.027f);
        std::vector<uint8_t> mask(rows, 1); mask[rows - 1] = 0;
        (void)cpu.forward(x, mask); (void)gpu.forward(x.to(Device::GPU), mask);
    }
    const float expected = cpu.accumulate_switch_aux_grad(0.07f);
    reset_gpu_transfer_stats();
    const Tensor loss = gpu.accumulate_switch_aux_grad_device(0.07f);
    const auto stats = gpu_transfer_stats();
    require(loss.get_device() == Device::GPU && loss.size == 1, "Switch aux scalar left device");
    require(stats.d2h_calls == 0 && stats.stream_synchronizations == 0,
            "Switch aux introduced a host boundary before objective aggregation");
    require(std::abs(expected - loss.cpu().data()[0]) < 2e-5f, "Switch aux deferred loss mismatch");
    for (size_t p = 0; p < cp.size(); ++p) {
        require((cp[p]->grad.size == 0) == (gp[p]->grad.size == 0), "Switch aux missing gradient");
        if (cp[p]->grad.size) assert_close(cp[p]->grad, gp[p]->grad, 2e-4f, "Switch aux deferred VJP", 2e-3f);
    }
    gpu.materialize_expert_loads();
    for (int e = 0; e < 5; ++e)
        require(std::abs(cpu.expert_loads[e] - gpu.expert_loads[e]) < 2e-4f,
                "Switch aux lost final accumulated load telemetry");
    // No residual records/loads may enter the next step, including empty masks.
    gpu.begin_aux_accumulation();
    (void)gpu.forward(Tensor::ones({3, 16}, Device::GPU), {0, 0, 0});
    reset_gpu_transfer_stats();
    const Tensor zero = gpu.accumulate_switch_aux_grad_device(0.07f);
    require(gpu_transfer_stats().d2h_calls == 0, "Empty aux objective downloaded payload");
    require(zero.cpu().data()[0] == 0, "Empty aux objective retained previous loss");
    gpu.begin_aux_accumulation();
    bool invalid = false;
    try { (void)gpu.accumulate_switch_aux_grad_device(std::numeric_limits<float>::quiet_NaN()); }
    catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "Switch aux accepted a nonfinite coefficient");
    // Wide expert count, stable ties, and both atomic/ordered reductions.
    Tensor probabilities({257, 65}, Device::CPU);
    for (int row = 0; row < 257; ++row) for (int e = 0; e < 65; ++e)
        probabilities.data()[row * 65 + e] = 1.0f / 65;
    const Tensor probabilities_gpu = probabilities.to(Device::GPU);
    float reference_loss = 0;
    const Tensor reference_grad = MoERouter::switch_aux_grad_logits(probabilities, 3, 0.07f, &reference_loss);
    for (bool deterministic : {false, true}) {
        determinism::set_deterministic_reductions(deterministic);
        Tensor device_loss;
        reset_gpu_transfer_stats();
        const Tensor device_grad = MoERouter::switch_aux_grad_logits_device(probabilities_gpu, 3, 0.07f, &device_loss);
        require(gpu_transfer_stats().d2h_calls == 0, "Pure Switch core downloaded scalar");
        require(std::abs(reference_loss - device_loss.cpu().data()[0]) < 2e-5f, "Wide Switch loss/tie mismatch");
        assert_close(reference_grad, device_grad, 1e-7f, "Wide Switch exact gradient", 1e-3f);
    }
    determinism::set_deterministic_reductions(true);
}
} // namespace

int main() {
    return run_parity("training_redesign", [] {
        set_strict_gpu_execution(true);
        determinism::set_deterministic_reductions(true);
        set_env("NSOS_MAMBA_BOUNDARY_HISTORY","1");
        set_env("NSOS_MAMBA_BACKWARD_CHUNK_SIZE","32");
        set_env("NSOS_MAMBA_CHUNKED_BACKWARD","1");
        set_env("NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD","1");
        set_env("NSOS_MAMBA_FORWARD_CHUNK_SIZE","32");
        for (int S : {1,33,65}) for (bool checkpoint : {false,true}) mamba_case(S,checkpoint);
        norm_case();
        moe_load_case();
        moe_aux_device_case(false);
        moe_aux_device_case(true);
        for (int window : {1,9,INT_MAX}) {
            attention_case(17,5,2,33,window);
            attention_case(33,4,2,64,window);
            attention_case(9,2,1,128,window);
        }
        set_env("NSOS_ATTN_TILED_TRAINING","1invalid");
        bool rejected=false;
        try { (void)training_policy::tiled_attention(); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"malformed training policy was accepted");
    });
}
