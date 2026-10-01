// SSA (Subquadratic Sparse Attention) CPU↔GPU parity.
//
// Guards the CUDA kernels in src/cuda/sparse_attention_kernels.cu: block-mean
// routing, sink+local+top-k block selection, and online-softmax exact
// attention over the selected positions.  Historically sparse_selective_attention
// ran host loops over `.data()` and could not run on a GPU tensor.  This test
// checks the GPU path matches the CPU reference both with the raw-query router
// and with a learned selector Wsel.

#include "gpu_parity_common.h"
#include "sparse_attention.h"
#include "tensor.h"

using nsos::Device;
using nsos::SparseAttentionConfig;
using nsos::Tensor;
using nsos::sparse_selective_attention;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

int main() {
  return run_parity("sparse_attention", [] {
    constexpr int n = 40;   // spans 5 blocks of 8 -> content selection is active
    constexpr int d = 16;

    SparseAttentionConfig cfg;
    cfg.block_size = 8;
    cfg.top_k_blocks = 2;
    cfg.local_blocks = 1;
    cfg.sink_blocks = 1;
    cfg.scale = 0.0f;  // -> 1/sqrt(d)

    Tensor Q = Tensor::random({n, d}, Device::CPU);
    Tensor K = Tensor::random({n, d}, Device::CPU);
    Tensor V = Tensor::random({n, d}, Device::CPU);

    // (1) Raw-query routing (Wsel == nullptr).
    const Tensor cpu_out =
        sparse_selective_attention(Q, K, V, cfg, nullptr, nullptr);
    const Tensor gpu_out =
        sparse_selective_attention(Q.to(Device::GPU), K.to(Device::GPU),
                                   V.to(Device::GPU), cfg, nullptr, nullptr)
            .cpu();
    cuda_sync_or_throw("sparse_attention/raw");
    assert_close(cpu_out, gpu_out, 2e-3f, "sparse_attention_raw");

    // (2) Learned selector routing (route = Wsel @ q).
    Tensor Wsel = Tensor::random({d, d}, Device::CPU);
    const Tensor cpu_sel =
        sparse_selective_attention(Q, K, V, cfg, nullptr, &Wsel);
    Tensor Wsel_gpu = Wsel.to(Device::GPU);
    const Tensor gpu_sel =
        sparse_selective_attention(Q.to(Device::GPU), K.to(Device::GPU),
                                   V.to(Device::GPU), cfg, nullptr, &Wsel_gpu)
            .cpu();
    cuda_sync_or_throw("sparse_attention/wsel");
    assert_close(cpu_sel, gpu_sel, 2e-3f, "sparse_attention_wsel");
  });
}
