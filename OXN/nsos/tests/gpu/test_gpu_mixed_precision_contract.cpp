// CUDA mixed-precision capability and execution contract.
//
// FP16 is executed on sm_70+, BF16 on sm_80+.  Unsupported requests must throw
// instead of silently falling back to FP32.  On capable GPUs the test also runs
// a real FP16 training step and validates dynamic loss-scaler advancement.

#include "gpu_parity_common.h"
#include "jamba.h"
#include "trainer.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using nsos::Device;
using nsos::JambaModel;
using nsos::ModelConfig;
using nsos::Tensor;
using nsos::Trainer;
using nsos::TrainPhaseScheduler;
using nsos::set_matmul_precision_mode;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool low_precision_matmul_is_rejected(const Tensor& a, const Tensor& b,
                                      int mode) {
  set_matmul_precision_mode(mode);
  try {
    (void)a.matmul(b);
    cuda_sync_or_throw("mixed_precision/unsupported_probe");
  } catch (const std::runtime_error& error) {
    set_matmul_precision_mode(0);
    const std::string message(error.what());
    const std::string expected =
        mode == 1 ? "requires sm_80" : "requires sm_70";
    if (message.find(expected) == std::string::npos) {
      throw std::runtime_error(
          "mixed-precision request failed for the wrong reason: " + message);
    }
    return true;
  }
  set_matmul_precision_mode(0);
  return false;
}

} // namespace

int main() {
  return run_parity("mixed_precision_contract", [] {
#ifdef USE_CUDA
    int device = 0;
    cudaDeviceProp properties{};
    require(cudaGetDevice(&device) == cudaSuccess,
            "cannot query active CUDA device");
    require(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
            "cannot query CUDA compute capability");
    std::cout << "[GPUParity:mixed_precision_contract] active_sm="
              << properties.major << properties.minor << std::endl;

    Tensor a_cpu({16, 16}, Device::CPU);
    Tensor b_cpu({16, 16}, Device::CPU);
    for (int index = 0; index < a_cpu.size; ++index) {
      a_cpu.data()[index] =
          static_cast<float>((index % 29) - 14) / 17.0f;
      b_cpu.data()[index] =
          static_cast<float>((index % 31) - 15) / 19.0f;
    }
    Tensor a_gpu = a_cpu.to(Device::GPU);
    Tensor b_gpu = b_cpu.to(Device::GPU);
    Tensor fp32_reference = a_cpu.matmul(b_cpu);

    if (properties.major >= 7) {
      set_matmul_precision_mode(2);
      Tensor fp16_output = a_gpu.matmul(b_gpu);
      cuda_sync_or_throw("mixed_precision/fp16_matmul");
      set_matmul_precision_mode(0);
      assert_close(fp16_output.cpu(), fp32_reference, 3e-2f,
                   "mixed_precision_fp16_matmul");

      ModelConfig config;
      config.num_layers = 1;
      config.d_model = 16;
      config.vocab_size = 32;
      config.n_heads = 4;
      config.n_kv_heads = 2;
      config.attention_period = 64;
      config.attention_slot = 63;
      config.mamba2_faithful = false;
      config.tie_word_embeddings = false;
      config.use_moe = false;
      config.use_ttt = false;
      config.use_chrass = false;
      config.use_kan = false;
      config.dropout = 0.0f;
      config.use_exact_attention_training = false;

      JambaModel model(config, Device::GPU);
      Trainer trainer(&model, 1e-3f);
      TrainPhaseScheduler schedule;
      schedule.progressive_qat_enabled = false;
      trainer.configure_progressive_qat(schedule);
      trainer.weight_decay = 0.0f;
      trainer.max_grad_norm = 1000.0f;
      trainer.loss_scale = 1024.0f;
      trainer.loss_scale_growth_interval = 1;
      set_matmul_precision_mode(2);
      const float loss =
          trainer.train_step({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6});
      cuda_sync_or_throw("mixed_precision/fp16_training");
      set_matmul_precision_mode(0);
      require(std::isfinite(loss), "FP16 GPU training loss is non-finite");
      require(trainer.loss_scale == 2048.0f &&
                  !trainer.last_optimizer_step_skipped,
              "FP16 GPU step did not advance the dynamic loss scaler");
      std::cout << "[GPUParity:mixed_precision_contract] fp16=executed"
                << " loss_scale=" << trainer.loss_scale << std::endl;
    } else {
      require(low_precision_matmul_is_rejected(a_gpu, b_gpu, 2),
              "pre-sm_70 GPU silently accepted FP16 Tensor-Core mode");
      std::cout << "[GPUParity:mixed_precision_contract] fp16=rejected_as_unsupported"
                << std::endl;
    }

    if (properties.major >= 8) {
      set_matmul_precision_mode(1);
      Tensor bf16_output = a_gpu.matmul(b_gpu);
      cuda_sync_or_throw("mixed_precision/bf16_matmul");
      set_matmul_precision_mode(0);
      assert_close(bf16_output.cpu(), fp32_reference, 6e-2f,
                   "mixed_precision_bf16_matmul");
      std::cout << "[GPUParity:mixed_precision_contract] bf16=executed"
                << std::endl;
    } else {
      require(low_precision_matmul_is_rejected(a_gpu, b_gpu, 1),
              "pre-sm_80 GPU silently accepted BF16 Tensor-Core mode");
      std::cout << "[GPUParity:mixed_precision_contract] bf16=rejected_as_unsupported"
                << std::endl;
    }
    set_matmul_precision_mode(0);
#endif
  });
}
