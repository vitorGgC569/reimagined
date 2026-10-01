// KAN (Kolmogorov-Arnold) layer CPU↔GPU parity.
//
// Guards the CUDA RBF-basis kernels added in Phase 2 (src/cuda/kan_kernels.cu):
// historically BitFastKANLayer::compute_basis and the backward RBF derivative
// ran host loops over `.data()`, so the layer could not run on a GPU tensor at
// all.  This test constructs CPU and GPU layers, weight-synchronises them, and
// checks forward output, grad_input, and every parameter gradient match.

#include "gpu_parity_common.h"
#include "kan.h"
#include "tensor.h"
#include "training_runtime_policy.h"
#include "gpu_execution.h"
#include "cuda/kan_kernels.cuh"
#include "cuda/moe_training_wmma.cuh"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <bit>

#include <vector>

using nsos::BitFastKANLayer;
using nsos::Device;
using nsos::Parameter;
using nsos::Tensor;
using nsos::gpu_parity_test::assert_close;
using nsos::gpu_parity_test::cuda_sync_or_throw;
using nsos::gpu_parity_test::run_parity;

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void set_kan_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_KAN_RECOMPUTE_TRAINING",enabled?"1":"0")==0,"cannot set KAN fixture policy");
#else
  require(setenv("NSOS_KAN_RECOMPUTE_TRAINING",enabled?"1":"0",1)==0,"cannot set KAN fixture policy");
#endif
}
void set_kan_wmma_policy(bool enabled) {
#ifdef _WIN32
  require(_putenv_s("NSOS_KAN_WMMA_TRAINING",enabled?"1":"0")==0,"cannot set KAN WMMA policy");
#else
  require(setenv("NSOS_KAN_WMMA_TRAINING",enabled?"1":"0",1)==0,"cannot set KAN WMMA policy");
#endif
}

// Byte-for-byte copy of every Parameter from `src` to `dst` (same ordering,
// since both layers are built with identical constructor arguments).
void mirror_parameters(const std::vector<Parameter*>& src,
                       const std::vector<Parameter*>& dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("mirror_parameters: parameter count mismatch");
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (!src[i] || !dst[i]) continue;
    if (src[i]->data.size != dst[i]->data.size) {
      throw std::runtime_error("mirror_parameters: size mismatch at index " +
                               std::to_string(i));
    }
    Tensor cpu_src = src[i]->data.get_device() == Device::GPU
                         ? src[i]->data.cpu()
                         : src[i]->data;
    if (dst[i]->data.get_device() == Device::GPU) {
      dst[i]->data.copy_from(cpu_src.to(Device::GPU));
    } else {
      dst[i]->data.copy_from(cpu_src);
    }
  }
}

void recompute_parity(int mode,bool quantized,int rows,int inputs,int outputs,int grid,bool batched,bool wmma=false) {
  set_kan_wmma_policy(false);
  set_kan_policy(false);
  nsos::set_matmul_precision_mode(mode);
  BitFastKANLayer legacy(inputs,outputs,grid), candidate(inputs,outputs,grid);
  for (auto* p:legacy.parameters()) for (int i=0;i<p->data.size;++i)
    p->data.data()[i]=0.11f*std::sin(0.071f*(i+1));
  legacy.to(Device::GPU); candidate.to(Device::GPU);
  mirror_parameters(legacy.parameters(),candidate.parameters());
  legacy.set_quantized(quantized); candidate.set_quantized(quantized);
  const std::vector<int> shape=batched?std::vector<int>{2,rows,inputs}:std::vector<int>{rows,inputs};
  Tensor input(shape,Device::CPU);
  for (int i=0;i<input.size;++i) input.data()[i]=0.7f*std::sin(0.039f*(i+1));
  auto gradient_shape=shape; gradient_shape.back()=outputs;
  Tensor gradient(gradient_shape,Device::CPU);
  for (int i=0;i<gradient.size;++i) gradient.data()[i]=0.03f*std::cos(0.047f*(i+1));
  Tensor x=input.to(Device::GPU), dy=gradient.to(Device::GPU);
  const Tensor expected=legacy.forward(x).cpu(), expected_dx=legacy.backward(dy).cpu();
  std::vector<Tensor> expected_gradients;
  for (auto* p:legacy.parameters()) expected_gradients.push_back(p->grad.cpu());
  set_kan_policy(true);
  set_kan_wmma_policy(wmma);
  const auto before=nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanWmmaGemm)];
  nsos::reset_gpu_transfer_stats();
  const Tensor result=candidate.forward(x);
  require(candidate.retained_basis_elements()==0,"recompute retained a materialized RBF basis");
  require(nsos::gpu_transfer_stats().d2h_calls==0,"KAN device QAT returned a scale to host");
  // Mutating caller input must not corrupt the owned recompute tape.
  x.copy_from(Tensor::zeros(shape,Device::GPU));
  const Tensor dx=candidate.backward(dy);
  require(nsos::gpu_transfer_stats().d2h_calls==0,"KAN recomputed backward downloaded a payload");
  if (wmma && moe_training_wmma_geometry(batched?rows*2:rows,inputs,outputs,mode))
    require(nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanWmmaGemm)]==before+3,
            "KAN WMMA did not execute projection and both VJPs");
  else require(nsos::gpu::dispatch_counters()[static_cast<unsigned>(nsos::gpu::DispatchPath::KanWmmaGemm)]==before,
               "KAN dispatched WMMA for an ineligible geometry");
  const float atol=mode==0?3e-5f:3e-4f;
  assert_close(expected,result,atol,"KAN implicit projection",3e-3f);
  assert_close(expected_dx,dx,atol,"KAN recomputed input gradient",4e-3f);
  auto params=candidate.parameters();
  for (size_t i=0;i<params.size();++i)
    assert_close(expected_gradients[i],params[i]->grad,atol,"KAN recomputed parameter gradient",4e-3f);
  bool rejected=false;
  try { candidate.backward(dy); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"KAN consumed its tape twice");
  std::cout<<"KAN recompute mode="<<mode<<" qat="<<quantized<<" rows="<<rows<<" dims="<<inputs<<","<<outputs<<" grid="<<grid<<std::endl;
  set_kan_policy(false);
  set_kan_wmma_policy(false);
}

// Independent double primal and VJP, no CPU/GPU KAN or BLAS reference.
// QAT oracle holds absmean/rounding detached and applies the declared STE mask.
void double_oracle(bool quantized,bool wmma=false) {
  const int R=wmma?17:3,D=wmma?33:5,O=wmma?35:4,G=5;
  auto operand=[&](double value) {
    if (!wmma) return value;
    // Independent host BF16 round-to-nearest-even, no GPU conversion/BLAS.
    uint32_t bits=std::bit_cast<uint32_t>(static_cast<float>(value));
    bits+=0x7fff+((bits>>16)&1);
    return static_cast<double>(std::bit_cast<float>(bits&0xffff0000U));
  };
  BitFastKANLayer layer(D,O,G);
  std::vector<double> bw(O*D),rw(O*D*G),bias(O),x(R*D),dy(R*O);
  auto initialize=[](std::vector<double>& v,int modulus,double denom) {
    for (size_t i=0;i<v.size();++i) v[i]=(static_cast<int>((i*7)%modulus)-modulus/2)/denom;
  };
  initialize(bw,15,32); initialize(rw,17,32); initialize(bias,11,128);
  initialize(x,13,8); initialize(dy,11,32);
  auto params=layer.parameters();
  for (int i=0;i<O*D;++i) params[0]->data.data()[i]=static_cast<float>(bw[i]);
  for (int i=0;i<O*D*G;++i) params[1]->data.data()[i]=static_cast<float>(rw[i]);
  for (int i=0;i<O;++i) params[2]->data.data()[i]=static_cast<float>(bias[i]);
  auto effective=[&](const std::vector<double>& latent,float& scale) {
    double sum=0; for (double v:latent) sum+=std::abs(v);
    scale=static_cast<float>(sum/latent.size())+1e-8f;
    std::vector<double> result=latent;
    if (quantized) for (size_t i=0;i<latent.size();++i) {
      const float value=static_cast<float>(latent[i])*(1.0f/(scale+1e-8f));
      result[i]=(value>0.5f?1.0f:value< -0.5f?-1.0f:0.0f)*scale;
    }
    return result;
  };
  float bs,rs;
  const auto eb=effective(bw,bs),er=effective(rw,rs);
  std::vector<double> y(R*O,0),gx(R*D,0),gb(O*D,0),gr(O*D*G,0),gbi(O,0);
  for (int r=0;r<R;++r) for (int o=0;o<O;++o) {
    double value=bias[o];
    gbi[o]+=dy[r*O+o];
    for (int d=0;d<D;++d) {
      value+=operand(x[r*D+d])*operand(eb[o*D+d]);
      gx[r*D+d]+=operand(dy[r*O+o])*operand(eb[o*D+d]);
      gb[o*D+d]+=operand(dy[r*O+o])*operand(x[r*D+d]);
      for (int g=0;g<G;++g) {
        const double center=-1+g*0.5, width=0.5, diff=(x[r*D+d]-center)/width;
        const double phi=std::exp(-0.5*diff*diff);
        const int wi=(o*D+d)*G+g;
        value+=operand(phi)*operand(er[wi]); gr[wi]+=operand(dy[r*O+o])*operand(phi);
        gx[r*D+d]+=operand(dy[r*O+o])*operand(er[wi])*phi*(center-x[r*D+d])/(width*width);
      }
    }
    y[r*O+o]=value;
  }
  if (quantized) {
    for (size_t i=0;i<gb.size();++i) if (std::fabs(static_cast<float>(bw[i])*(1.0f/(bs+1e-8f)))>1) gb[i]=0;
    for (size_t i=0;i<gr.size();++i) if (std::fabs(static_cast<float>(rw[i])*(1.0f/(rs+1e-8f)))>1) gr[i]=0;
  }
  layer.set_quantized(quantized); layer.to(Device::GPU);
  Tensor input({R,D},Device::CPU),gradient({R,O},Device::CPU);
  for (int i=0;i<input.size;++i) input.data()[i]=static_cast<float>(x[i]);
  for (int i=0;i<gradient.size;++i) gradient.data()[i]=static_cast<float>(dy[i]);
  set_kan_policy(true); set_kan_wmma_policy(wmma); nsos::set_matmul_precision_mode(wmma?1:0);
  const Tensor actual_y=layer.forward(input.to(Device::GPU)).cpu();
  const Tensor actual_x=layer.backward(gradient.to(Device::GPU)).cpu();
  auto check=[](const Tensor& value,const std::vector<double>& expected) {
    const Tensor host=value.cpu();
    for (int i=0;i<host.size;++i) require(std::isfinite(host.data()[i]) &&
      std::abs(host.data()[i]-expected[i])<4e-6*(1+std::abs(expected[i])),"KAN independent double oracle differs");
  };
  check(actual_y,y); check(actual_x,gx); params=layer.parameters();
  check(params[0]->grad,gb); check(params[1]->grad,gr); check(params[2]->grad,gbi);
  if (!quantized && !wmma) {
    auto primal=[&](const std::vector<double>& values,const std::vector<double>& b,
                    const std::vector<double>& w,const std::vector<double>& bi) {
      double loss=0;
      for (int r=0;r<R;++r) for (int o=0;o<O;++o) {
        double value=bi[o];
        for (int d=0;d<D;++d) {
          value+=values[r*D+d]*b[o*D+d];
          for (int g=0;g<G;++g) {
            const double diff=(values[r*D+d]-(-1+g*0.5))/0.5;
            value+=std::exp(-0.5*diff*diff)*w[(o*D+d)*G+g];
          }
        }
        loss+=value*dy[r*O+o];
      }
      return loss;
    };
    for (int group=0;group<4;++group) {
      auto& values=group==0?x:group==1?bw:group==2?rw:bias;
      const auto& expected=group==0?gx:group==1?gb:group==2?gr:gbi;
      for (size_t i=0;i<values.size();++i) {
        const double old=values[i], eps=1e-5;
        values[i]=old+eps; const double plus=primal(x,bw,rw,bias);
        values[i]=old-eps; const double minus=primal(x,bw,rw,bias);
        values[i]=old;
        require(std::abs((plus-minus)/(2*eps)-expected[i])<1e-8,"KAN double finite difference oracle failed");
      }
    }
  }
  set_kan_policy(false);
  set_kan_wmma_policy(false);
}

void tape_guards() {
  nsos::set_matmul_precision_mode(0); set_kan_policy(true);
  BitFastKANLayer layer(5,7,3); layer.to(Device::GPU);
  Tensor x=Tensor::ones({3,5},Device::GPU),dy=Tensor::ones({3,7},Device::GPU);
  for (int mutation=0;mutation<6;++mutation) {
    layer.forward(x);
    if (mutation==0) ++layer.base_weight.version;
    if (mutation==1) layer.set_quantized(true);
    if (mutation==2) nsos::set_matmul_precision_mode(1);
    if (mutation==3) set_kan_policy(false);
    if (mutation==4) set_kan_wmma_policy(true);
    if (mutation==5) layer.base_weight.data.shape.dims[1]=4;
    bool rejected=false;
    try { layer.backward(dy); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"KAN accepted a stale compute/parameter tape");
    require(layer.base_weight.grad.size==0,"KAN tape rejection published gradients");
    layer.base_weight.data.shape.dims[1]=5;
    layer.set_quantized(false); nsos::set_matmul_precision_mode(0); set_kan_policy(true); set_kan_wmma_policy(false);
  }
  layer.forward(x);
  bool rejected=false;
  try { layer.backward(Tensor::ones({1,3,7},Device::GPU)); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"KAN accepted a flattened but rank-incompatible gradient");
  layer.to(Device::GPU);
  rejected=false;
  try { layer.backward(dy); } catch (const std::runtime_error&) { rejected=true; }
  require(rejected,"KAN device transition kept a pending tape");
  set_kan_policy(false);
}

void qat_scale_oracle() {
  constexpr int N=1041;
  Tensor w({N},Device::CPU);
  double sum=0;
  for (int i=0;i<N;++i) { w.data()[i]=(i%31-15)/32.0f; sum+=std::abs(w.data()[i]); }
  const float expected=static_cast<float>(sum/N)+1e-8f;
  const Tensor weight=w.to(Device::GPU);
  Tensor effective=Tensor::uninitialized({N},Device::GPU), scale=Tensor::uninitialized({1},Device::GPU),
      partials=Tensor::uninitialized({(N-1)/256+1},Device::GPU);
  nsos::reset_gpu_transfer_stats();
  require(nsos::cuda::launch_kan_prepare_ternary(weight.raw_data(),effective.raw_data(),scale.raw_data(),partials.raw_data(),N),
          "KAN QAT fixture launch failed");
  require(nsos::gpu_transfer_stats().d2h_calls==0,"KAN QAT oracle launch downloaded scale");
  const Tensor actual_scale=scale.cpu(), actual_weight=effective.cpu();
  require(actual_scale.data()[0]==expected,"KAN ordered QAT scale differs from independent absmean");
  for (int i=0;i<N;++i) {
    const float v=w.data()[i]*(1.0f/(expected+1e-8f));
    require(actual_weight.data()[i]==(v>0.5f?1.0f:v< -0.5f?-1.0f:0.0f)*expected,
            "KAN ordered QAT ternary threshold differs");
  }
  require(nsos::cuda::launch_kan_prepare_ternary(weight.raw_data(),effective.raw_data(),scale.raw_data(),partials.raw_data(),N),
          "KAN QAT repeat failed");
  const Tensor repeat=effective.cpu();
  for (int i=0;i<N;++i) require(repeat.data()[i]==actual_weight.data()[i],"KAN QAT preparation is nondeterministic");
  require(!nsos::cuda::launch_kan_rbf_projection(weight.raw_data(),weight.raw_data(),scale.raw_data(),scale.raw_data(),
          weight.raw_data(),weight.raw_data(),effective.raw_data(),1,std::numeric_limits<int>::max(),1,2,0),
          "KAN overflow geometry was accepted");
  bool rejected=false;
  try {
    nsos::cuda::launch_kan_rbf_basis_forward(weight.raw_data(),scale.raw_data(),scale.raw_data(),
        effective.raw_data(),1,std::numeric_limits<int>::max()/2+1,2);
  } catch (const std::overflow_error&) { rejected=true; }
  require(rejected,"KAN legacy basis indexing overflow was accepted");
  rejected=false;
  try {
    nsos::cuda::launch_kan_rbf_basis_backward(weight.raw_data(),weight.raw_data(),scale.raw_data(),
        scale.raw_data(),effective.raw_data(),1,std::numeric_limits<int>::max()/2+1,2);
  } catch (const std::overflow_error&) { rejected=true; }
  require(rejected,"KAN legacy gradient basis indexing overflow was accepted");
}

}  // namespace

int main() {
  return run_parity("kan", [] {
    set_kan_wmma_policy(false);
    set_kan_policy(false);
    constexpr int in_features = 24;
    constexpr int out_features = 32;
    constexpr int grid = 5;
    constexpr int batch = 2;
    constexpr int seq = 4;

    BitFastKANLayer cpu_layer(in_features, out_features, grid);
    BitFastKANLayer gpu_layer(in_features, out_features, grid);
    gpu_layer.to(Device::GPU);
    mirror_parameters(cpu_layer.parameters(), gpu_layer.parameters());

    Tensor input = Tensor::random({batch, seq, in_features}, Device::CPU);

    // Forward parity.
    const Tensor cpu_out = cpu_layer.forward(input);
    const Tensor gpu_out = gpu_layer.forward(input.to(Device::GPU)).cpu();
    cuda_sync_or_throw("kan/forward");
    assert_close(cpu_out, gpu_out, 2e-3f, "kan_forward");

    // Backward parity: grad_input plus every parameter gradient.
    Tensor grad = Tensor::random({batch, seq, out_features}, Device::CPU);
    const Tensor cpu_gi = cpu_layer.backward(grad);
    const Tensor gpu_gi = gpu_layer.backward(grad.to(Device::GPU)).cpu();
    cuda_sync_or_throw("kan/backward");
    assert_close(cpu_gi, gpu_gi, 2e-3f, "kan_grad_input");

    const auto cpu_params = cpu_layer.parameters();
    const auto gpu_params = gpu_layer.parameters();
    for (size_t i = 0; i < cpu_params.size(); ++i) {
      assert_close(cpu_params[i]->grad, gpu_params[i]->grad, 2e-3f,
                   "kan_param_grad");
    }
    for (int mode:{0,1,2}) for (bool qat:{false,true}) {
      recompute_parity(mode,qat,17,33,35,5,false);
      recompute_parity(mode,qat,3,7,9,3,true);
    }
    double_oracle(false); double_oracle(true); tape_guards(); qat_scale_oracle();
    if (moe_training_wmma_supported()) for (int mode:{0,1,2}) for (bool qat:{false,true}) {
      recompute_parity(mode,qat,17,33,35,5,false,true);
      recompute_parity(mode,qat,33,65,67,3,false,true);
      recompute_parity(mode,qat,3,7,9,3,true,true);
    }
    if (moe_training_wmma_supported()) { double_oracle(false,true); double_oracle(true,true); }
    nsos::set_matmul_precision_mode(0);
  });
}
