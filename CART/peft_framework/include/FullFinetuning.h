#ifndef FULL_FINETUNING_H
#define FULL_FINETUNING_H

#include "Tensor.h"
#include <memory>

class FullFinetuningLayer {
public:
    FullFinetuningLayer(int input_dim, int output_dim);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& upstream_grad);

    const Tensor* get_grad_W() const { return m_grad_W.get(); }
    const Tensor* get_grad_B() const { return m_grad_B.get(); }
    const Tensor* get_W() const { return m_weights.get(); }
    const Tensor* get_B() const { return m_bias.get(); }

private:
    int m_input_dim;
    int m_output_dim;
    std::unique_ptr<Tensor> m_weights;
    std::unique_ptr<Tensor> m_bias;
    std::unique_ptr<Tensor> m_last_input;
    std::unique_ptr<Tensor> m_grad_W;
    std::unique_ptr<Tensor> m_grad_B;
};

#endif // FULL_FINETUNING_H
