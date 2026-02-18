#ifndef KAN_H
#define KAN_H

#include "Tensor.h"
#include <memory>
#include <vector>

class Spline {
public:
    Spline(int order = 3);
    float eval(float x) const;
    std::vector<float> m_control_points;
};

class KANLayer {
public:
    KANLayer(int input_dims, int output_dims, int spline_order);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& upstream_grad);

//public for testing in research mode
//private:
    std::unique_ptr<Tensor> m_last_input; // Stored for backward pass
    // Gradients for each spline's control points
    std::vector<std::vector<std::vector<float>>> m_spline_grads;

    int m_input_dims;
    int m_output_dims;
    int m_spline_order;
    std::vector<std::vector<Spline>> m_splines;
};

#endif  // KAN_H
