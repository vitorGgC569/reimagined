#ifndef IA3_H
#define IA3_H

#include "Tensor.h"
#include <memory>

class IA3Layer {
public:
    IA3Layer(int input_dim, int output_dim);
    void setBaseWeights(const Tensor& weights);
    void enablePureScalingMode(bool enable); // New method to avoid identity matrix
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& upstream_grad); // Modified to return dL/dX
    void update(float learning_rate);

    const Tensor* get_grad_L() const { return m_grad_L.get(); }
    const Tensor* get_L() const { return m_L.get(); }

private:
    int m_input_dim;
    int m_output_dim;
    bool m_pure_scaling_mode = false; // Flag for scaler mode
    std::unique_ptr<Tensor> m_L; // Learned scaling vector
    std::unique_ptr<Tensor> m_W0; // Fixed base weights
    std::unique_ptr<Tensor> m_grad_L;
    std::unique_ptr<Tensor> m_last_input;
    std::unique_ptr<Tensor> m_pre_scale_output; // Intermediate output before scaling
};

#endif // IA3_H
