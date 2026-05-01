#ifndef DORA_H
#define DORA_H

#include "Tensor.h"
#include <memory>

class DoRALayer {
public:
    DoRALayer(int input_dims, int output_dims, int rank);
    void setBaseWeights(const Tensor& weights);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& upstream_grad);
    void update(float learning_rate);

    const Tensor* get_grad_A() const { return m_grad_A.get(); }
    const Tensor* get_grad_B() const { return m_grad_B.get(); }
    const Tensor* get_grad_m() const { return m_grad_m.get(); }
    const Tensor* get_m() const { return m_m.get(); }

private:
    int m_input_dims;
    int m_output_dims;
    int m_rank;
    std::unique_ptr<Tensor> m_W0;
    std::unique_ptr<Tensor> m_A;
    std::unique_ptr<Tensor> m_B;
    std::unique_ptr<Tensor> m_m; // Magnitude vector

    std::unique_ptr<Tensor> m_grad_A;
    std::unique_ptr<Tensor> m_grad_B;
    std::unique_ptr<Tensor> m_grad_m;

    std::unique_ptr<Tensor> m_last_input;
    std::unique_ptr<Tensor> m_V_norm; // Normalized weights cache
    std::unique_ptr<Tensor> m_V;      // Un-normalized weights cache
};

#endif // DORA_H
