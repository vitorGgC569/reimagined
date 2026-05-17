#pragma once
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "nsos_config.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace nsos {

struct TensorShape {
    std::vector<int> dims;
    std::vector<size_t> strides;

    TensorShape() = default;
    TensorShape(const std::vector<int>& d) : dims(d) {
        compute_strides();
    }
    
    void compute_strides() {
        if (dims.empty()) {
            strides.clear();
            return;
        }
        strides.resize(dims.size());
        size_t s = 1;
        for (int i = (int)dims.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= dims[i];
        }
    }

    size_t numel() const {
        if (dims.empty()) return 1;
        size_t n = 1;
        for (int d : dims) n *= d;
        return n;
    }

    size_t size() const { return dims.size(); }
    int operator[](int i) const { return dims[i]; }
    int back() const { return dims.empty() ? 0 : dims.back(); }
    bool empty() const { return dims.empty(); }
    
    std::vector<int>::const_iterator begin() const { return dims.begin(); }
    std::vector<int>::const_iterator end() const { return dims.end(); }
    bool operator==(const TensorShape& other) const { return dims == other.dims; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }
};

struct TensorDeleter {
    Device device;
    TensorDeleter(Device dev) : device(dev) {}
    void operator()(float* ptr);
};

class Tensor {
public:
    std::shared_ptr<float> data_ptr;
    TensorShape shape;
    int size;
    Device device;
    std::shared_ptr<Tensor> grad;

    Tensor();
    Tensor(std::vector<int> s, Device dev = Device::CPU, float fill_value = 0.0f);
    
    // Raw pointer access.  When `eager_gpu_sync_enabled()` is true
    // AND the tensor lives on GPU, an implicit cudaDeviceSynchronize
    // runs first so the caller observes a consistent view of UM.
    //
    // This is the dynamic safety net for the Pascal+Windows UM bug:
    // setting NSOS_EAGER_GPU_SYNC=1 turns ALL data() calls into a
    // sync barrier on GPU, eliminating the entire class of "host
    // access on UM with pending kernel" segfaults at the cost of one
    // cudaDeviceSynchronize per access (typically dominated by
    // pipeline serialization, not driver overhead).
    //
    // Default is OFF — kernel-launch hot paths that pass data() into
    // launch_xxx_kernel() do not need a sync because kernels on the
    // default stream serialize with each other.  Set the env var when
    // running on Pascal+Windows, or call sync_host_access() at known
    // CPU-access sites for a more targeted fix.
    float* data();
    const float* data() const;

    // Returns the raw pointer without any implicit sync.  Use ONLY
    // when passing into a CUDA kernel launch (which serializes with
    // prior launches via the default stream and does not need host-
    // side synchronization).
    float* raw_data() { return data_ptr.get(); }
    const float* raw_data() const { return data_ptr.get(); }

    // Explicit synchronization barrier.  Equivalent to calling data()
    // when eager sync is enabled.  Idempotent and cheap (~1µs) when
    // the GPU has no pending work.
    void sync_host_access() const;

    // Process-wide toggle for eager GPU sync inside data().  Reads
    // NSOS_EAGER_GPU_SYNC env var on first call.  Safe to call from
    // any thread (uses an atomic flag).
    static bool eager_gpu_sync_enabled();

    Tensor to(Device dev) const;
    Tensor cpu() const;
    Tensor clone() const;
    void copy_from(const Tensor& other);
    
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor transpose() const;
    
    Tensor relu() const;
    // LEARN S1 (BitNet b1.58 2B4T): Squared ReLU activation.
    //   forward:  y = max(0, x)^2
    //   backward: dx = dy * 2 * max(0, x)
    // Used as the FFN/MoE activation because SwiGLU under low-precision
    // (ternary BitLinear / FP8) can spike and overflow the dynamic
    // range, causing loss divergence after extended training.  Squared
    // ReLU is numerically stable in quantized regimes and gives
    // comparable expressive power to SwiGLU at moderate parameter
    // counts (1-2B params, validated by BitNet b1.58 2B4T technical
    // report 2026).
    Tensor squared_relu() const;
    // Backward for squared_relu.  Returns dx = dy * 2 * max(0, pre).
    // pre_activation is the value BEFORE squared_relu was applied
    // (typically saved during forward).  dy is the upstream gradient.
    static Tensor squared_relu_backward(const Tensor& dy,
                                         const Tensor& pre_activation);
    Tensor sigmoid() const;
    Tensor softmax(int dim = -1) const;
    Tensor rmsnorm(float eps = 1e-6f) const;
    Tensor clamp(float min, float max) const;
    
    Tensor sum(int dim = -1, bool keepdim = false) const;
    float norm() const;
    
    Tensor rmsnorm_backward(const Tensor& grad, const Tensor& x_norm) const;
    std::pair<float, Tensor> cross_entropy(const std::vector<int>& target) const;
    std::pair<float, Tensor> mse_loss(const Tensor& target) const;
    
    Tensor reshape(std::vector<int> new_shape) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor slice(int dim, int start, int end) const;
    int get_flat_index(const std::vector<int>& indices) const;
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    float get(const std::vector<int>& indices) const;
    
    void print(const std::string& name = "", int max_elements = 10) const;
    
    // Static Factories with Overloads for Vector support
    static Tensor from_blob(void* ptr, std::vector<int> s, Device dev, bool take_ownership=false);
    static Tensor zeros(const std::vector<int>& s, Device dev = Device::CPU) { return Tensor(s, dev, 0.0f); }
    static Tensor zeros(const TensorShape& s, Device dev = Device::CPU) { return Tensor(s.dims, dev, 0.0f); }
    static Tensor ones(const std::vector<int>& s, Device dev = Device::CPU) { return Tensor(s, dev, 1.0f); }
    static Tensor ones(const TensorShape& s, Device dev = Device::CPU) { return Tensor(s.dims, dev, 1.0f); }
    static Tensor random(const std::vector<int>& s, Device dev = Device::CPU);
    static Tensor kaiming_uniform(const std::vector<int>& s, Device dev = Device::CPU);
    static Tensor xavier_uniform(const std::vector<int>& s, Device dev = Device::CPU);
    static Tensor from_scalar(float val, Device dev = Device::CPU);
    static Tensor eye(int n, Device dev = Device::CPU);
    
    static void clip_grad_norm_(std::vector<Tensor>& params, float max_norm);

    uintptr_t data_ptr_int() const { return reinterpret_cast<uintptr_t>(data()); }
    Device get_device() const { return device; }

    void zero_grad() {
        if (!grad) return;
#ifdef USE_CUDA
        if (grad->get_device() == Device::GPU) {
            cudaMemset(grad->data(), 0, static_cast<size_t>(grad->size) * sizeof(float));
            cudaDeviceSynchronize();
            return;
        }
#endif
        std::fill_n(grad->data(), grad->size, 0.0f);
    }
    void add_grad(const Tensor& g) {
        if (!grad) grad = std::make_shared<Tensor>(Tensor::zeros(shape.dims, device));
        Tensor new_grad = grad->add(g);
        grad->copy_from(new_grad);
    }
};

} // namespace nsos
