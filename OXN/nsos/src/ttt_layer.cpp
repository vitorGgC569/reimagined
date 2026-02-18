#include "../include/ttt_layer.h"
#include "../include/nsos_arena.h"
#include <cmath>
#include <iostream>
#include <random>

namespace nsos {

TTTLayer::TTTLayer(int dim, int hidden, float lr) 
    : dim(dim), hidden(hidden), learning_rate(lr),
      w_k(std::make_unique<BitLinear>(dim, hidden)),
      w_v(std::make_unique<BitLinear>(dim, hidden)),
      w_out(std::make_unique<BitLinear>(hidden, dim)) {
      
      momentum_ = Tensor::zeros({hidden, dim}, Device::CPU);
      grad_accum_ = Tensor::zeros({hidden, dim}, Device::CPU);
}

void TTTLayer::initialize_from_meta(const Tensor &x, const Tensor &y) {
    float eps = learning_rate;
    float friction = 0.9f;
    
    Tensor force = compute_force(x, y);
    
    // Kick-Drift- Kick (Leapfrog)
    momentum_ = momentum_.mul(friction).add(force.mul(0.5f * eps));
    
    // Update Weights of w_out (Position)
    // Accessing internal parameter data
    auto params = w_out->parameters();
    if(!params.empty()) {
        params[0]->data = params[0]->data.add(momentum_.mul(eps));
    }
    
    Tensor new_force = compute_force(x, y);
    momentum_ = momentum_.mul(friction).add(new_force.mul(0.5f * eps));
}

Tensor TTTLayer::compute_force(const Tensor& x, const Tensor& target) {
    Tensor k = w_k->forward(x);
    Tensor pred = w_out->forward(k);
    Tensor diff = pred.sub(target);
    // Negative Gradient as Force
    return k.transpose().matmul(diff).mul(-1.0f);
}

Tensor TTTLayer::forward(const Tensor &x) {
    Tensor k = w_k->forward(x);
    return w_out->forward(k);
}

Tensor TTTLayer::backward(const Tensor &g) { return g; }
void TTTLayer::reset() { momentum_ = Tensor::zeros({hidden, dim}, Device::CPU); }
void TTTLayer::to(Device d) { w_k->to(d); w_v->to(d); w_out->to(d); }

std::vector<Parameter*> TTTLayer::parameters() { 
    std::vector<Parameter*> res;
    auto p1 = w_k->parameters(); res.insert(res.end(), p1.begin(), p1.end());
    auto p2 = w_v->parameters(); res.insert(res.end(), p2.begin(), p2.end());
    auto p3 = w_out->parameters(); res.insert(res.end(), p3.begin(), p3.end());
    return res;
}

void TTTLayer::set_use_hamiltonian(bool) {}
void TTTLayer::set_temperature(float) {}
void TTTLayer::set_friction(float) {}
void TTTLayer::set_max_grad_norm(float) {}
void TTTLayer::set_checkpoint_interval(int) {}
Tensor TTTLayer::get_current_adaptation() { return momentum_; }

} // namespace nsos
