#pragma once
#include "autograd.h"
#include "bitlinear.h"
#include <vector>
#include <memory>

namespace nsos {

struct TTTSessionSnapshot {
    bool has_state = false;
    Tensor momentum;
    Tensor adaptation;
};

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

    TTTLayer(int dim, int hidden, float lr = 0.001f, uint64_t seed = 0);
    
    Tensor forward(const Tensor &x);
    Tensor backward(const Tensor &g);
    
    void initialize_from_meta(const Tensor &x, const Tensor &y);
    void reset();
    void to(Device d);
    std::vector<Parameter*> parameters();
    void collect_bitlinear_layers(std::vector<BitLinear*>& out);
    void set_training_mode(bool enabled);
    
    void set_use_hamiltonian(bool enabled);
    void set_temperature(float t);
    void set_friction(float f);
    void set_max_grad_norm(float m);
    void set_checkpoint_interval(int i);
    
    Tensor get_current_adaptation();
    TTTSessionSnapshot snapshot_state() const;
    void restore_state(const TTTSessionSnapshot& snapshot);

private:
    bool training_mode_ = true;
    bool use_hamiltonian_ = false;
    float temperature_ = 1.0f;
    float friction_ = 0.9f;
    float max_grad_norm_ = 1.0f;
    int checkpoint_interval_ = 16;

    Tensor saved_input_;
    Tensor saved_keys_;
    Tensor saved_values_;
    Tensor saved_pre_adaptation_;
    Tensor saved_errors_;

    Tensor compute_force(const Tensor& x, const Tensor& target);
};

} // namespace nsos
