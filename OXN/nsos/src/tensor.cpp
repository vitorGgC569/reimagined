#include "../include/tensor.h"
#include "../include/tensor_iterator.h"
#include "../include/nsos_arena.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <random>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nsos {

void TensorDeleter::operator()(float* ptr) {
    // Arena handles free
}

Tensor::Tensor() : size(0), device(Device::CPU) {
    shape = TensorShape({});
    data_ptr = nullptr;
}

Tensor::Tensor(std::vector<int> s, Device dev, float fill_value) : device(dev) {
    shape = TensorShape(s);
    size = (int)shape.numel();
    if (size == 0) {
        data_ptr = nullptr;
        return;
    }
    float* raw_ptr = (float*)ArenaAllocator::instance().alloc(size * sizeof(float), dev);
    if (!raw_ptr) throw std::runtime_error("OOM");
    if (dev == Device::CPU) {
        if (fill_value == 0.0f) std::memset(raw_ptr, 0, size * sizeof(float));
        else std::fill_n(raw_ptr, size, fill_value);
    }
    data_ptr = std::shared_ptr<float>(raw_ptr, TensorDeleter(dev));
}

Tensor Tensor::random(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    float* d = t.data();
    for(int i=0; i<t.size; ++i) d[i] = dist(gen);
    return t;
}

Tensor Tensor::kaiming_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    float fan_in = (float)s[0];
    float bound = std::sqrt(6.0f / fan_in);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* d = t.data();
    for(int i=0; i<t.size; ++i) d[i] = dist(gen);
    return t;
}

Tensor Tensor::xavier_uniform(const std::vector<int>& s, Device dev) {
    Tensor t(s, dev);
    float fan_in = (float)s[0];
    float fan_out = (float)(s.size() > 1 ? s[1] : s[0]);
    float bound = std::sqrt(6.0f / (fan_in + fan_out));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-bound, bound);
    float* d = t.data();
    for(int i=0; i<t.size; ++i) d[i] = dist(gen);
    return t;
}

Tensor Tensor::eye(int n, Device dev) {
    Tensor t = zeros({n, n}, dev);
    float* d = t.data();
    for(int i=0; i<n; ++i) d[i*n + i] = 1.0f;
    return t;
}

Tensor Tensor::from_scalar(float val, Device dev) {
    return ones({1}, dev).mul(val);
}

void Tensor::clip_grad_norm_(std::vector<Tensor>& params, float max_norm) {
    float total_norm = 0.0f;
    for(auto& p : params) {
        float n = p.norm();
        total_norm += n * n;
    }
    total_norm = std::sqrt(total_norm);
    if(total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for(auto& p : params) {
            float* d = p.data();
            for(int i=0; i<p.size; ++i) d[i] *= scale;
        }
    }
}

Tensor Tensor::add(const Tensor& other) const {
    TensorShape out_s;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_s);
    Tensor res(out_s.dims, device); 
    TensorIterator iter(res, *this, other);
    iter.parallel_for_each([](float a, float b) { return a + b; });
    return res;
}

Tensor Tensor::sub(const Tensor& other) const {
    TensorShape out_s;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_s);
    Tensor res(out_s.dims, device);
    TensorIterator iter(res, *this, other);
    iter.parallel_for_each([](float a, float b) { return a - b; });
    return res;
}

Tensor Tensor::mul(const Tensor& other) const {
    TensorShape out_s;
    TensorIterator::compute_broadcast_shape(shape, other.shape, out_s);
    Tensor res(out_s.dims, device);
    TensorIterator iter(res, *this, other);
    iter.parallel_for_each([](float a, float b) { return a * b; });
    return res;
}

Tensor Tensor::mul(float scalar) const {
    Tensor res(shape.dims, device);
    const float* src = data();
    float* dst = res.data();
    #pragma omp parallel for
    for(int i=0; i<size; ++i) dst[i] = src[i] * scalar;
    return res;
}

Tensor Tensor::matmul(const Tensor& other) const {
    int rank = (int)shape.size();
    int M = shape[rank-2];
    int K = shape[rank-1];
    int N = other.shape.back();
    std::vector<int> out_dims = shape.dims;
    out_dims.back() = N;
    Tensor res(out_dims, device);
    return res; 
}

Tensor Tensor::rmsnorm(float eps) const {
    Tensor res(shape.dims, device);
    int inner = shape.back();
    int outer = size / inner;
    const float* src = data();
    float* dst = res.data();
    #pragma omp parallel for
    for(int i=0; i<outer; ++i) {
        float ss = 0.0f;
        for(int j=0; j<inner; ++j) ss += src[i*inner+j]*src[i*inner+j];
        float inv = 1.0f / std::sqrt(ss/inner + eps);
        for(int j=0; j<inner; ++j) dst[i*inner+j] = src[i*inner+j]*inv;
    }
    return res;
}

float Tensor::norm() const {
    float s = 0.0f;
    const float* d = data();
    for(int i=0; i<size; ++i) s += d[i]*d[i];
    return std::sqrt(s);
}

Tensor Tensor::clone() const {
    Tensor t(shape.dims, device);
    std::memcpy(t.data(), data(), size*sizeof(float));
    return t;
}

Tensor Tensor::to(Device dev) const {
    if(device == dev) return *this;
    Tensor res(shape.dims, dev);
    return res;
}

Tensor Tensor::cpu() const { return to(Device::CPU); }

void Tensor::copy_from(const Tensor& other) {
    if(size != other.size) throw std::runtime_error("Size mismatch");
    std::memcpy(data(), other.data(), size*sizeof(float));
}

Tensor Tensor::reshape(std::vector<int> s) const {
    Tensor t = *this; t.shape = TensorShape(s); return t;
}

Tensor Tensor::transpose(int d0, int d1) const { return clone(); }
Tensor Tensor::transpose() const { return transpose(-2, -1); }
Tensor Tensor::softmax(int dim) const { return clone(); }
Tensor Tensor::sum(int d, bool k) const { return ones({1}, device); }
Tensor Tensor::slice(int d, int s, int e) const { return clone(); }
Tensor Tensor::clamp(float min, float max) const { return clone(); }
Tensor Tensor::relu() const { return clone(); }
Tensor Tensor::sigmoid() const { return clone(); }
Tensor Tensor::rmsnorm_backward(const Tensor& g, const Tensor& x) const { return g; }
std::pair<float, Tensor> Tensor::cross_entropy(const std::vector<int>& t) const { return {0.0f, Tensor()}; }
std::pair<float, Tensor> Tensor::mse_loss(const Tensor& t) const { return {0.0f, Tensor()}; }
void Tensor::print(const std::string& n, int m) const {}

Tensor Tensor::from_blob(void* p, std::vector<int> s, Device d, bool o) {
    Tensor t; t.shape = TensorShape(s); t.size = (int)t.shape.numel(); t.device = d;
    t.data_ptr = std::shared_ptr<float>((float*)p, [](float*){});
    return t;
}

} // namespace nsos
