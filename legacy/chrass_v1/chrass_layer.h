#pragma once
#include "tensor.h"
#include <vector>
#include <string>

namespace nsos {

class ChrassLayer {
public:
    int dim;
    Tensor bias;
    
    // CSR Sparse Matrix Representation
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    
    // Optimizer State (AdamW)
    std::vector<float> m;
    std::vector<float> v;
    std::vector<float> grad_values;
    int t;

    ChrassLayer(int dimension, const std::vector<float>& adjacency);
    
    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad_output, const Tensor& input);
    void step(float lr);
    
    // Serialization
    std::vector<float> to_dense();
};

} // namespace nsos
