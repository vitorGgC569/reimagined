#ifndef SSM_H
#define SSM_H

#include "Tensor.h"
#include <memory>

class SSMLayer {
public:
    SSMLayer(int state_dim, int input_dim);
    Tensor forward(const Tensor& sequence);
    void backward(const Tensor& upstream_grad_sequence);

    // [TEST ONLY / dev] Getters for weights and grads
    Tensor* get_A() { return m_A.get(); }
    Tensor* get_B() { return m_B.get(); }
    Tensor* get_C() { return m_C.get(); }
    Tensor* get_grad_A() { return m_grad_A.get(); }


private:
    // Needed for BPTT
    std::vector<std::unique_ptr<Tensor>> m_history_x;
    std::vector<std::unique_ptr<Tensor>> m_history_h;

    std::unique_ptr<Tensor> m_grad_A;
    std::unique_ptr<Tensor> m_grad_B;
    std::unique_ptr<Tensor> m_grad_C;

    int m_state_dim;
    int m_input_dim;

    std::unique_ptr<Tensor> m_A;
    std::unique_ptr<Tensor> m_B;
    std::unique_ptr<Tensor> m_C;
    std::unique_ptr<Tensor> m_h;
};

#endif // SSM_H
