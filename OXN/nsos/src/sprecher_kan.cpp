#include "../include/sprecher_kan.h"

namespace nsos {

SprecherKAN::SprecherKAN(int in, int out, int h) : in_dim(in), out_dim(out), hidden_dim(h) {
    // Init
}

Tensor SprecherKAN::forward(const Tensor& x) { return Tensor::zeros({x.shape[0], out_dim}); }
Tensor SprecherKAN::backward(const Tensor& g) { return Tensor::zeros({g.shape[0], in_dim}); }
void SprecherKAN::to(Device d) {}
void SprecherKAN::extend_grid(int new_grid) {}

} // namespace nsos
