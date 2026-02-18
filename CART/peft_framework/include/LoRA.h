#ifndef LORA_H
#define LORA_H

#include "Tensor.h"
#include <memory>

class LoRALayer {
public:
    LoRALayer(int input_dims, int output_dims, int rank);
    LoRALayer(std::unique_ptr<Tensor> A, std::unique_ptr<Tensor> B);
    virtual ~LoRALayer() = default;

    void setBaseWeights(const Tensor& weights);
    virtual Tensor forward(const Tensor& input);
    void backward(const Tensor& upstream_grad);
    void update(float learning_rate);

    const Tensor* get_grad_A() const { return m_grad_A.get(); }
    const Tensor* get_grad_B() const { return m_grad_B.get(); }
    const Tensor* get_A() const { return m_A.get(); }
    const Tensor* get_B() const { return m_B.get(); }
    int get_rank() const { return m_rank; }

protected:
    std::unique_ptr<Tensor> m_W0;
    std::unique_ptr<Tensor> m_A;
    std::unique_ptr<Tensor> m_B;
    std::unique_ptr<Tensor> m_grad_A;
    std::unique_ptr<Tensor> m_grad_B;
    std::unique_ptr<Tensor> m_last_input;
    int m_rank;
};

#endif // LORA_H
