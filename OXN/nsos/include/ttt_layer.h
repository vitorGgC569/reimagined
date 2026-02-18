#pragma once
#include "autograd.h"
#include "bitlinear.h"
#include <vector>
#include <memory>

namespace nsos {

class TTTLayer {
public:
    int dim;
    int hidden;
    float learning_rate;
    
    std::unique_ptr<BitLinear> w_k;
    std::unique_ptr<BitLinear> w_v;
    std::unique_ptr<BitLinear> w_out;
    
    Tensor momentum_;
    Tensor grad_accum_;
    
    uint64_t state_;
    float cached_gaussian_;

    TTTLayer(int dim, int hidden, float lr = 0.001f);
    
    Tensor forward(const Tensor &x);
    Tensor backward(const Tensor &g);
    
    void initialize_from_meta(const Tensor &x, const Tensor &y);
    void reset();
    void to(Device d);
    std::vector<Parameter*> parameters();
    
    void set_use_hamiltonian(bool enabled);
    void set_temperature(float t);
    void set_friction(float f);
    void set_max_grad_norm(float m);
    void set_checkpoint_interval(int i);
    
    Tensor get_current_adaptation();

private:
    Tensor compute_force(const Tensor& x, const Tensor& target);
};

} // namespace nsos
