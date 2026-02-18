#ifndef DYNAMIC_CHRASS_H
#define DYNAMIC_CHRASS_H

#include "tensor.h"
#include <cmath>
#include <map>
#include <vector>


namespace nsos {

class DynamicCHRASS {
  std::map<int, std::vector<int>> adjacency;
  std::map<std::pair<int, int>, float> weights;
  int dim;
  float plasticity_rate;

public:
  DynamicCHRASS(int dimension, float rate = 0.01f)
      : dim(dimension), plasticity_rate(rate) {}

  Tensor forward_hebbian(const Tensor &x) {
    int batch = x.shape[0];
    Tensor output = Tensor::zeros({batch, dim}, x.get_device());
    float *out_ptr = output.data();
    const float *in_ptr = x.data();

    for (auto const &[connection, weight] : weights) {
      int src = connection.first;
      int dst = connection.second;
      out_ptr[dst] += weight * in_ptr[src];
    }

    for (auto &[connection, weight] : weights) {
      int src = connection.first;
      int dst = connection.second;
      float pre = in_ptr[src];
      float post = out_ptr[dst];
      float delta = plasticity_rate * (pre * post - post * post * weight);
      weight += delta;
      if (std::abs(weight) < 1e-5f) {
        weight = 0.0f;
      }
    }
    return output;
  }

  void grow_synapse(int src, int dst, float init_weight) {
    if (src < dim && dst < dim) {
      adjacency[src].push_back(dst);
      weights[{src, dst}] = init_weight;
    }
  }

  int count_synapses() const { return static_cast<int>(weights.size()); }
};
} // namespace nsos

#endif
