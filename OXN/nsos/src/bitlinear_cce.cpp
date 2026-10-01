#include "bitlinear.h"
#include "bitnet_gpu_dispatch.h"
#include "hadamard.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nsos {
TiledCrossEntropyResult BitLinear::cross_entropy_tiled(const Tensor& input,const TiledCrossEntropyOptions& o) {
    if(!training_mode_ || input.size==0 || input.shape.size()<2 || input.shape.back()!=in_features ||
       input.size/in_features>std::numeric_limits<int>::max() || out_features>16777216 ||
       o.row_tile<1 || o.row_tile>4096 || o.vocabulary_tile<1 || o.vocabulary_tile>4096 ||
       !std::isfinite(o.gradient_scale) || o.gradient_scale<=0 || !std::isfinite(o.l2_beta) || o.l2_beta<0 ||
       !std::isfinite(o.repetition_scale) || o.repetition_scale<0)
        throw std::invalid_argument("Invalid tiled head training contract");
    const int rows=static_cast<int>(input.size/in_features); const Device dev=input.get_device();
    if(o.targets.size()!=static_cast<size_t>(rows) || o.weights.size()!=static_cast<size_t>(rows) ||
       (!o.l2_weights.empty() && o.l2_weights.size()!=static_cast<size_t>(rows)) ||
       (o.repetition_scale>0 && (o.sequence_length<1 || rows%o.sequence_length!=0)) ||
       weight.data.shape.dims!=std::vector<int>{out_features,in_features} || weight.data.get_device()!=dev ||
       (!exact_linear_mode_ && (magnitude.data.size!=out_features || magnitude.data.get_device()!=dev)) ||
       (use_bias && (bias.data.size!=out_features || bias.data.get_device()!=dev)))
        throw std::invalid_argument("Tiled head metadata/parameter mismatch");
    for(int r=0;r<rows;++r) {
        if(o.targets[r]<-1 || o.targets[r]>=out_features || !std::isfinite(o.weights[r]) || o.weights[r]<0 ||
           (o.targets[r]<0 && o.weights[r]!=0) || (!o.l2_weights.empty() && (!std::isfinite(o.l2_weights[r]) || o.l2_weights[r]<0)))
            throw std::invalid_argument("Invalid tiled head target or row weight");
    }
    const uint64_t version=weight.version;
    const bool cpu_packed=!use_reference_path && dev==Device::CPU;
    const bool gpu_qat=!use_reference_path && dev==Device::GPU && !quantization_sensitive_ && !loqa.active;
    if(!use_reference_path && dev==Device::GPU && !gpu_qat)
        throw std::invalid_argument("Tiled GPU head requires reference or supported QAT BitLinear backward");
    discard_backward_state();
    Tensor effective=weight.data, qat_scale;
    if(gpu_qat) effective=qat_fake_quant_ternary_absmean(weight.data,&qat_scale);
    else if(cpu_packed) {
        if(!packed_weight_valid || packed_weight_version!=weight.version) repack_weights();
        effective=Tensor::uninitialized({out_features,in_features},dev);
        if(unpacked_weights_i8.size()!=static_cast<size_t>(effective.size)) throw std::logic_error("CCE packed weight inventory mismatch");
        for(size_t i=0;i<static_cast<size_t>(effective.size);++i) effective.data()[i]=unpacked_weights_i8[i]*weight_scale;
    } else if(!use_reference_path) effective=materialize_weight_for_device(dev);
    Tensor dw=Tensor::zeros({out_features,in_features},dev);
    Tensor dm=Tensor::zeros({1,out_features},dev), db=Tensor::zeros({1,out_features},dev);
    Tensor da, dadapter_b;
    if(loqa.active) {da=Tensor::zeros(loqa.A.data.shape.dims,dev);dadapter_b=Tensor::zeros(loqa.B.data.shape.dims,dev);}
    TiledCrossEntropyResult result;
    result.losses=Tensor::zeros({3},dev);result.input_gradient=Tensor::zeros({rows,in_features},dev);
    Tensor flat=input.reshape({rows,in_features});
    for(int row=0;row<rows;row+=o.row_tile) {
        const int count=std::min(o.row_tile,rows-row);
        Tensor meta_host({count,7},Device::CPU);
        for(int r=0;r<count;++r) {
            int global=row+r;float* m=meta_host.data()+r*7;
            m[0]=static_cast<float>(o.targets[global]);m[1]=o.weights[global];m[2]=(o.l2_beta==0 || o.l2_weights.empty())?0:o.l2_weights[global];
            for(int k=0;k<4;++k)m[3+k]=-1;
            if(o.repetition_scale>0 && o.targets[global]>=0) {
                int local=global%o.sequence_length, sample=global-local, n=0;
                for(int previous=local-1;previous>=std::max(0,local-4);--previous) {
                    int id=o.targets[sample+previous];
                    if(id<0 || id==o.targets[global] || id==o.eos_token)continue;
                    bool duplicate=false;for(int k=0;k<n;++k)duplicate|=static_cast<int>(m[3+k])==id;
                    if(!duplicate && n<4)m[3+n++]=static_cast<float>(id);
                }
            }
        }
        Tensor meta=meta_host.to(dev);
        Tensor raw=flat.storage_view(static_cast<size_t>(row)*in_features,{count,in_features}).clone();
        cce::mask_inactive_rows(raw,meta);
        const bool normalized=norm_strategy==NormStrategy::RMS_PERI || norm_strategy==NormStrategy::RMS_PRE;
        Tensor norm=normalized?raw.rmsnorm(1e-6f):raw;
        Tensor linear=norm;
        std::vector<float> scales;
        if(gpu_qat) linear=qat_fake_quant_activations(norm,precision_bits);
        else if(cpu_packed) {
            linear=norm.clone();if(use_hadamard)hadamard_transform(linear.data(),count,in_features);
            Tensor codes=quantize_activations_bitnet(linear,scales);
            for(int r=0;r<count;++r)for(int k=0;k<in_features;++k)linear.data()[r*in_features+k]=codes.data()[r*in_features+k]*scales[r];
        }
        Tensor adapter_input;if(loqa.active)adapter_input=norm.matmul(loqa.A.data);
        Tensor stat_host=Tensor::zeros({count,12},Device::CPU);
        for(int r=0;r<count;++r)stat_host.data()[r*12]=-std::numeric_limits<float>::infinity();
        Tensor stat=stat_host.to(dev);
        auto project=[&](int v,int end,Tensor* pre_out) {
            Tensor pre=matmul_nt(linear,effective.storage_view(static_cast<size_t>(v)*in_features,{end-v,in_features}));
            if(loqa.active)pre=pre.add(adapter_input.matmul(loqa.B.data.slice(1,v,end)));
            if(pre_out)*pre_out=pre;
            Tensor z=exact_linear_mode_?pre:pre.mul(magnitude.data.storage_view(v,{end-v}));
            if(use_bias)z=z.add(bias.data.storage_view(v,{end-v}));return z;
        };
        for(int v=0;v<out_features;v+=o.vocabulary_tile) {
            Tensor z=project(v,std::min(v+o.vocabulary_tile,out_features),nullptr);
            result.maximum_logit_elements=std::max(result.maximum_logit_elements,static_cast<size_t>(z.size));
            cce::update_statistics(z,stat,meta,v);
        }
        result.losses.add_inplace_(cce::finish_statistics(stat,meta,out_features,o.repetition_scale,o.l2_beta));
        Tensor dx=Tensor::zeros({count,in_features},dev), adapter_dx=Tensor::zeros({count,in_features},dev);
        for(int v=0;v<out_features;v+=o.vocabulary_tile) {
            int end=std::min(v+o.vocabulary_tile,out_features);Tensor pre;Tensor z=project(v,end,&pre);
            Tensor g=cce::gradient_tile(z,stat,meta,v,out_features,o.repetition_scale,o.l2_beta,o.gradient_scale);
            if(use_bias)cce::add_region(db,g.sum(0).reshape({1,end-v}),0,v);
            if(!exact_linear_mode_)cce::add_region(dm,g.mul(pre).sum(0).reshape({1,end-v}),0,v);
            Tensor gp=exact_linear_mode_?g:g.mul(magnitude.data.storage_view(v,{end-v}));
            cce::add_region(dw,matmul_tn(gp,linear),v,0);
            dx.add_inplace_(gp.matmul(effective.storage_view(static_cast<size_t>(v)*in_features,{end-v,in_features})));
            if(loqa.active) {
                cce::add_region(dadapter_b,matmul_tn(adapter_input,gp),0,v);
                Tensor gb=matmul_nt(gp,loqa.B.data.slice(1,v,end));
                da.add_inplace_(matmul_tn(norm,gb));adapter_dx.add_inplace_(matmul_nt(gb,loqa.A.data));
            }
        }
        if(cpu_packed && use_hadamard)hadamard_transform(dx.data(),count,in_features);
        if(loqa.active)dx.add_inplace_(adapter_dx);
        if(normalized)dx=raw.rmsnorm_backward(dx,norm,1e-6f);
        cce::add_region(result.input_gradient,dx,row,0);
    }
    if(weight.version!=version) throw std::logic_error("Head weight changed during CCE recomputation");
    if(gpu_qat)qat_ste_clip_weight_grad_device_scale(dw,weight.data,qat_scale);
    else if(cpu_packed) {
        for(size_t i=0;i<static_cast<size_t>(dw.size);++i) {
            if(use_tequila && std::abs(weight.data.data()[i])<0.05f)dw.data()[i]*=1.5f;
            if(std::abs(weight.data.data()[i]/(weight_scale+1e-8f))>1)dw.data()[i]=0;
        }
    }
    weight.add_grad(dw);
    if(!exact_linear_mode_)magnitude.add_grad(dm.reshape({out_features}));
    if(use_bias)bias.add_grad(db.reshape({out_features}));
    if(loqa.active){loqa.A.add_grad(da);loqa.B.add_grad(dadapter_b);}
    result.input_gradient=result.input_gradient.reshape(input.shape.dims);
    return result;
}
} // namespace nsos
