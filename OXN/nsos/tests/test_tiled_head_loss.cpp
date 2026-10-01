#include "bitlinear.h"
#include "jamba.h"
#include "trainer.h"
#include "runtime_execution_identity.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef USE_CUDA
#include "gpu_backend.h"
#include "gpu_execution.h"
#endif
using namespace nsos;
namespace {
void require(bool c,const char* msg){if(!c)throw std::runtime_error(msg);}
void policy(bool c) {
#ifdef _WIN32
    require(_putenv_s("NSOS_HEAD_CCE",c?"1":"0")==0,"cannot set CCE policy");
#else
    require(setenv("NSOS_HEAD_CCE",c?"1":"0",1)==0,"cannot set CCE policy");
#endif
}
void close(float a,float b,float tol=3e-4f){if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tol*(1+std::abs(a)))
    throw std::runtime_error("CCE numeric mismatch: "+std::to_string(a)+" vs "+std::to_string(b));}
void tensor_close(const Tensor& a,const Tensor& b,float tol=3e-4f){require(a.shape==b.shape,"CCE shape mismatch");
    Tensor x=a.cpu(),y=b.cpu();for(size_t i=0;i<static_cast<size_t>(x.size);++i)close(x.data()[i],y.data()[i],tol);}
std::pair<Tensor,Tensor> oracle(const Tensor& input,const TiledCrossEntropyOptions& o) {
    Tensor z=input.cpu();int rows=static_cast<int>(z.size/z.shape.back()),v=z.shape.back();
    Tensor g=Tensor::zeros(z.shape.dims),loss=Tensor::zeros({3});
    for(int r=0;r<rows;++r) {
        const float* zr=z.data()+r*v;float* gr=g.data()+r*v;
        double mx=*std::max_element(zr,zr+v),sum=0;std::vector<double> p(v);
        for(int j=0;j<v;++j){p[j]=std::exp(zr[j]-mx);sum+=p[j];}
        for(int j=0;j<v;++j)p[j]/=sum;
        if(o.targets[r]>=0) {loss.data()[0]+=static_cast<float>(o.weights[r]*(mx+std::log(sum)-zr[o.targets[r]]));
            for(int j=0;j<v;++j)gr[j]+=static_cast<float>(o.weights[r]*(p[j]-(j==o.targets[r])));}
        if(o.l2_beta>0 && !o.l2_weights.empty())for(int j=0;j<v;++j) {
            loss.data()[2]+=0.5f*o.l2_beta*o.l2_weights[r]*zr[j]*zr[j]/v;
            gr[j]+=o.l2_beta*o.l2_weights[r]*zr[j]/v;
        }
        if(o.repetition_scale>0 && o.targets[r]>=0) {
            int local=r%o.sequence_length;std::vector<int> seen;
            for(int k=local-1;k>=std::max(0,local-4);--k) {
                int n=o.targets[r-local+k];if(n<0 || n==o.targets[r] || n==o.eos_token || std::find(seen.begin(),seen.end(),n)!=seen.end())continue;
                seen.push_back(n);if(p[n]<=1e-6 || p[n]>=1-1e-6)continue;
                double factor=o.repetition_scale*p[n]/(1-p[n]);loss.data()[1]-=static_cast<float>(o.repetition_scale*std::log1p(-p[n]));
                for(int j=0;j<v;++j)gr[j]-=static_cast<float>(factor*p[j]);gr[n]+=static_cast<float>(factor);
            }
        }
        for(int j=0;j<v;++j)gr[j]*=o.gradient_scale;
    }
    return {loss,g.to(input.get_device())};
}
void linear_case(Device dev,bool exact,bool quantized,bool adapters,bool cpu_transforms=false) {
    std::cout << "linear case device=" << static_cast<int>(dev) << " exact=" << exact
              << " quantized=" << quantized << " adapters=" << adapters
              << " transforms=" << cpu_transforms << std::endl;
    BitLinear dense(8,17,true,421),tiled(8,17,true,421);
    dense.set_exact_linear_mode(exact);tiled.set_exact_linear_mode(exact);
    if(cpu_transforms) {
        dense.set_use_hadamard(true);tiled.set_use_hadamard(true);
        dense.set_use_tequila(true);tiled.set_use_tequila(true);
    }
    if(adapters) {
        Tensor a({8,3}),b({3,17});for(size_t i=0;i<static_cast<size_t>(a.size);++i)a.data()[i]=0.03f*std::sin(static_cast<float>(i));
        for(size_t i=0;i<static_cast<size_t>(b.size);++i)b.data()[i]=0.02f*std::cos(static_cast<float>(i));
        dense.loqa.A.data=a.clone();tiled.loqa.A.data=a.clone();dense.loqa.B.data=b.clone();tiled.loqa.B.data=b.clone();
        dense.set_use_loqa(true);tiled.set_use_loqa(true);
    }
    for(int i=0;i<17;++i){dense.magnitude.data.data()[i]=tiled.magnitude.data.data()[i]=0.5f+i*0.01f;
        dense.bias.data.data()[i]=tiled.bias.data.data()[i]=(i-8)*0.04f;}
    dense.set_reference_path(!quantized);tiled.set_reference_path(!quantized);dense.to(dev);tiled.to(dev);
    Tensor host({2,5,8});for(size_t i=0;i<static_cast<size_t>(host.size);++i)host.data()[i]=0.12f*std::sin(static_cast<float>(i)+0.4f);
    Tensor x=host.to(dev);TiledCrossEntropyOptions o;
    o.targets={-1,1,2,1,3,4,5,-1,4,6};o.weights={0,.25f,.5f,.25f,.125f,.2f,.4f,0,.2f,.3f};
    o.l2_weights=o.weights;o.sequence_length=5;o.eos_token=16;o.repetition_scale=.07f;o.l2_beta=.03f;o.gradient_scale=8;
    o.row_tile=3;o.vocabulary_tile=5;
    Tensor logits=dense.forward(x);auto expected=oracle(logits,o);Tensor dx=dense.backward(expected.second);
    auto result=tiled.cross_entropy_tiled(x,o);
    tensor_close(expected.first,result.losses);tensor_close(dx,result.input_gradient);
    tensor_close(dense.weight.grad,tiled.weight.grad);
    tensor_close(dense.bias.grad,tiled.bias.grad);
    if(!exact)tensor_close(dense.magnitude.grad,tiled.magnitude.grad);
    if(adapters){tensor_close(dense.loqa.A.grad,tiled.loqa.A.grad);tensor_close(dense.loqa.B.grad,tiled.loqa.B.grad);}
    require(result.maximum_logit_elements<=15,"CCE allocated an unbounded logits plane");
    // All ignored rows, including poison input, must produce exact zeros.
    Tensor poison({3,8});std::fill_n(poison.data(),24,std::numeric_limits<float>::quiet_NaN());
    TiledCrossEntropyOptions ignored;ignored.targets.assign(3,-1);ignored.weights.assign(3,0);ignored.l2_weights.assign(3,1);ignored.row_tile=2;ignored.vocabulary_tile=3;
    for(auto* p:tiled.parameters())p->zero_grad();
    auto empty=tiled.cross_entropy_tiled(poison.to(dev),ignored);Tensor eh=empty.input_gradient.cpu();
    for(int i=0;i<eh.size;++i)require(eh.data()[i]==0,"CCE ignored-row poison escaped");
    Tensor lh=empty.losses.cpu();for(int i=0;i<3;++i)require(lh.data()[i]==0,"CCE empty mask loss is not zero");
    for(auto* p:tiled.parameters())if(p->grad.size>0){Tensor gh=p->grad.cpu();for(int i=0;i<gh.size;++i)require(gh.data()[i]==0,"CCE empty mask gradient is not zero");}
}
void extreme_logits(Device dev) {
    BitLinear dense(3,7,true,912),tiled(3,7,true,912);
    dense.set_exact_linear_mode(true);tiled.set_exact_linear_mode(true);
    for(int i=0;i<7;++i)dense.bias.data.data()[i]=tiled.bias.data.data()[i]=-10000.f+static_cast<float>(i);
    dense.to(dev);tiled.to(dev);
    Tensor x=Tensor::zeros({2,3},dev);
    TiledCrossEntropyOptions o;o.targets={0,6};o.weights={.5f,.5f};o.row_tile=1;o.vocabulary_tile=2;
    auto expected=oracle(dense.forward(x),o);auto result=tiled.cross_entropy_tiled(x,o);
    tensor_close(expected.first,result.losses,2e-5f);
    dense.backward(expected.second);tensor_close(dense.bias.grad,tiled.bias.grad,2e-5f);
}
void finite_difference() {
    BitLinear layer(3,7,true,144);layer.set_exact_linear_mode(true);
    Tensor x({2,3});for(int i=0;i<6;++i)x.data()[i]=.1f*(i-2);
    TiledCrossEntropyOptions o;o.targets={1,6};o.weights={.3f,.7f};o.row_tile=1;o.vocabulary_tile=2;
    auto a=layer.cross_entropy_tiled(x,o);Tensor analytic=layer.weight.grad.clone();
    for(int i=0;i<21;++i) {
        float original=layer.weight.data.data()[i];const float eps=1e-3f;
        layer.weight.data.data()[i]=original+eps;layer.weight.mark_updated();
        float plus=layer.cross_entropy_tiled(x,o).losses.data()[0];
        layer.weight.data.data()[i]=original-eps;layer.weight.mark_updated();
        float minus=layer.cross_entropy_tiled(x,o).losses.data()[0];
        layer.weight.data.data()[i]=original;layer.weight.mark_updated();
        close(analytic.data()[i],(plus-minus)/(2*eps),2e-4f);
    }
    bool rejected=false;o.targets[0]=7;try{layer.cross_entropy_tiled(x,o);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"CCE accepted invalid target");
    // Malformed public primitive inputs must fail before indexing metadata.
    Tensor malformed=Tensor::zeros({2,11}),metadata=Tensor::zeros({2,7});
    rejected=false;try{cce::finish_statistics(malformed,metadata,7,0,0);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"CCE finish accepted malformed statistics");
}
void model_case(Device dev,bool supervised) {
    ModelConfig cfg;cfg.num_layers=1;cfg.d_model=16;cfg.vocab_size=37;cfg.n_heads=4;cfg.n_kv_heads=2;
    cfg.attention_period=64;cfg.use_moe=false;cfg.use_ttt=false;cfg.use_chrass=false;cfg.dropout=0;cfg.mamba2_faithful=false;
    cfg.tie_word_embeddings=true;cfg.logit_l2_beta=.02f;
    JambaModel dense(cfg,dev),tiled(cfg,dev);
    auto dp=dense.parameters(),tp=tiled.parameters();require(dp.size()==tp.size(),"CCE model registry mismatch");
    for(size_t i=0;i<dp.size();++i){tp[i]->data.copy_from(dp[i]->data);tp[i]->mark_updated();}
    Trainer d(&dense,.001f),t(&tiled,.001f);d.weight_decay=t.weight_decay=.01f;
    d.repetition_unlikelihood_scale=t.repetition_unlikelihood_scale=supervised?.03f:0;
    policy(false);float ld=supervised?d.train_supervised_batch({{1,2},{3,4,5}},{{6,7,6},{8,9}}):d.train_step({1,2,3,4},{2,3,4,5});
    const std::string dense_identity = d.execution_identity_digest();
    policy(true);float lt=supervised?t.train_supervised_batch({{1,2},{3,4,5}},{{6,7,6},{8,9}}):t.train_step({1,2,3,4},{2,3,4,5});
    close(ld,lt,6e-4f);for(size_t i=0;i<dp.size();++i)tensor_close(dp[i]->data,tp[i]->data,8e-4f);
    require(t.execution_identity_digest()!=dense_identity,"CCE policy absent from execution identity");
    policy(false);
}
}
int main(int argc,char** argv) {
    try {
        bool gpu=argc>1 && std::string(argv[1])=="--gpu";Device dev=gpu?Device::GPU:Device::CPU;
#ifdef USE_CUDA
        if(gpu){auto devices=gpu::enumerate_devices();require(!devices.empty(),"CCE GPU test requires real device");
            gpu::ExecutionContext lane;gpu::ExecutionContext::Scope scope(lane,true);
            linear_case(dev,false,false,false);linear_case(dev,true,true,false);linear_case(dev,false,true,false);
            linear_case(dev,false,false,true);model_case(dev,false);model_case(dev,true);}
        else
#else
        require(!gpu,"CCE GPU test reached a CPU build");
#endif
        {linear_case(dev,true,false,false);linear_case(dev,false,false,false);linear_case(dev,false,true,false);
         linear_case(dev,false,true,true,true);
         linear_case(dev,false,false,true);finite_difference();model_case(dev,false);model_case(dev,true);}
        extreme_logits(dev);
        std::cout<<"Tiled head loss: independent oracle, gradients, masks, STE, tying and Trainer passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
