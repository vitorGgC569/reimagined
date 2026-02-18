#pragma once
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "nsos_config.h"

namespace nsos {

struct TensorShape {
    std::vector<int> dims;
    std::vector<size_t> strides;

    TensorShape() = default;
    TensorShape(const std::vector<int>& d) : dims(d) {
        compute_strides();
    }
    
    void compute_strides() {
        if (dims.empty()) return;
        strides.resize(dims.size());
        size_t s = 1;
        for (int i = (int)dims.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= dims[i];
        }
    }

    size_t numel() const {
        if (dims.empty()) return 0;
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
    
    float* data() { return data_ptr.get(); }
    const float* data() const { return data_ptr.get(); }
    
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

    void zero_grad() { if (grad) std::fill_n(grad->data(), grad->size, 0.0f); }
    void add_grad(const Tensor& g) {
        if (!grad) grad = std::make_shared<Tensor>(Tensor::zeros(shape.dims, device));
        Tensor new_grad = grad->add(g);
        grad->copy_from(new_grad);
    }
};

} // namespace nsos
