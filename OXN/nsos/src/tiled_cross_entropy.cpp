#include "tiled_cross_entropy.h"
#include "cuda/tiled_cross_entropy.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nsos::cce {
namespace {
void validate_statistics(const Tensor& statistics, const Tensor& metadata) {
    if (statistics.shape.size()!=2 || statistics.shape[1]!=12 ||
        metadata.shape.dims!=std::vector<int>{statistics.shape[0],7} ||
        statistics.get_device()!=metadata.get_device())
        throw std::invalid_argument("CCE statistics/metadata shape or device mismatch");
}
void validate(const Tensor& logits, const Tensor& statistics, const Tensor& metadata) {
    validate_statistics(statistics,metadata);
    if (logits.shape.size()!=2 || statistics.shape.dims!=std::vector<int>{logits.shape[0],12} ||
        metadata.shape.dims!=std::vector<int>{logits.shape[0],7} ||
        statistics.get_device()!=logits.get_device() || metadata.get_device()!=logits.get_device())
        throw std::invalid_argument("CCE tile/statistics/metadata shape or device mismatch");
}
}
void mask_inactive_rows(Tensor& input,const Tensor& metadata) {
    if(input.shape.size()!=2 || metadata.shape.dims!=std::vector<int>{input.shape[0],7} ||
       input.get_device()!=metadata.get_device()) throw std::invalid_argument("CCE row mask mismatch");
#ifdef USE_CUDA
    if(input.get_device()==Device::GPU) {launch_cce_mask(input.raw_data(),metadata.raw_data(),input.shape[0],input.shape[1]);return;}
#endif
    for(int r=0;r<input.shape[0];++r) if(metadata.data()[r*7+1]==0 && metadata.data()[r*7+2]==0 && metadata.data()[r*7+3]<0)
        std::fill_n(input.data()+static_cast<size_t>(r)*input.shape[1],input.shape[1],0.0f);
}
void update_statistics(const Tensor& logits, Tensor& statistics, const Tensor& metadata, int start) {
    validate(logits,statistics,metadata);
    const int rows=logits.shape[0], columns=logits.shape[1];
#ifdef USE_CUDA
    if(logits.get_device()==Device::GPU) {
        launch_cce_statistics(logits.raw_data(),statistics.raw_data(),metadata.raw_data(),rows,columns,start);
        return;
    }
#endif
    const float* z=logits.data(); const float* m=metadata.data(); float* s=statistics.data();
    for(int r=0;r<rows;++r) {
        const float* mr=m+r*7; float* sr=s+r*12;
        if(mr[1]==0 && mr[2]==0 && mr[3]<0) continue;
        float mx=sr[0];
        for(int c=0;c<columns;++c) mx=std::max(mx,z[r*columns+c]);
        double sum=sr[1]==0 ? 0 : sr[1]*std::exp(static_cast<double>(sr[0]-mx));
        double sq=sr[7];
        for(int c=0;c<columns;++c) {
            const float v=z[r*columns+c]; const int id=start+c;
            sum+=std::exp(static_cast<double>(v-mx)); sq+=static_cast<double>(v)*v;
            if(id==static_cast<int>(mr[0])) sr[2]=v;
            for(int k=0;k<4;++k) if(id==static_cast<int>(mr[3+k])) sr[3+k]=v;
        }
        sr[0]=mx; sr[1]=static_cast<float>(sum); sr[7]=static_cast<float>(sq);
    }
}
Tensor finish_statistics(Tensor& statistics,const Tensor& metadata,int vocab,float rul,float beta) {
    validate_statistics(statistics,metadata);
    if(vocab<=0 || !std::isfinite(rul) || rul<0 || !std::isfinite(beta) || beta<0)
        throw std::invalid_argument("CCE finish coefficient or vocabulary mismatch");
    const int rows=statistics.shape[0]; Tensor losses=Tensor::zeros({rows,3},statistics.get_device());
#ifdef USE_CUDA
    if(statistics.get_device()==Device::GPU) {
        launch_cce_finish(statistics.raw_data(),losses.raw_data(),metadata.raw_data(),rows,vocab,rul,beta);
        return losses.sum(0);
    }
#endif
    float* s=statistics.data(); const float* m=metadata.data(); float* l=losses.data();
    for(int r=0;r<rows;++r) {
        float* sr=s+r*12; const float* mr=m+r*7;
        if(sr[1]==0) continue;
        if(mr[1]>0) l[r*3]=mr[1]*((sr[0]-sr[2])+std::log(sr[1]));
        if(beta>0 && mr[2]>0) l[r*3+2]=0.5f*beta*mr[2]*sr[7]/vocab;
        for(int k=0;k<4;++k) {
            const float p=mr[3+k]<0 ? 0 : std::exp(sr[3+k]-sr[0])/sr[1];
            sr[3+k]=0;
            if(p>1e-6f && p<1.0f-1e-6f) {
                sr[3+k]=rul*p/std::max(1.0f-p,1e-6f);
                l[r*3+1]-=rul*std::log1p(-p);
            }
        }
    }
    return losses.sum(0);
}
Tensor gradient_tile(const Tensor& logits,const Tensor& statistics,const Tensor& metadata,
                     int start,int vocab,float rul,float beta,float scale) {
    validate(logits,statistics,metadata); Tensor g=Tensor::zeros(logits.shape.dims,logits.get_device());
#ifdef USE_CUDA
    if(logits.get_device()==Device::GPU) {
        launch_cce_gradient(logits.raw_data(),g.raw_data(),statistics.raw_data(),metadata.raw_data(),
                            logits.shape[0],logits.shape[1],start,vocab,rul,beta,scale);
        return g;
    }
#endif
    const float* z=logits.data(); const float* s=statistics.data(); const float* m=metadata.data(); float* out=g.data();
    for(int r=0;r<logits.shape[0];++r) {
        const float* sr=s+r*12; const float* mr=m+r*7;
        if(sr[1]==0) continue;
        float factor=0; for(int k=0;k<4;++k) factor+=sr[3+k];
        for(int c=0;c<logits.shape[1];++c) {
            int id=start+c; int ix=r*logits.shape[1]+c;
            float p=std::exp(z[ix]-sr[0])/sr[1];
            float value=mr[1]*(p-(id==static_cast<int>(mr[0]) ? 1.0f : 0.0f))-factor*p;
            for(int k=0;k<4;++k) if(id==static_cast<int>(mr[3+k])) value+=sr[3+k];
            value+=beta*mr[2]*z[ix]/vocab; out[ix]=scale*value;
        }
    }
    return g;
}
void add_region(Tensor& destination,const Tensor& source,int row,int col) {
    if(destination.shape.size()!=2 || source.shape.size()!=2 || row<0 || col<0 ||
       row>destination.shape[0]-source.shape[0] || col>destination.shape[1]-source.shape[1] ||
       destination.get_device()!=source.get_device()) throw std::invalid_argument("CCE gradient region mismatch");
#ifdef USE_CUDA
    if(destination.get_device()==Device::GPU) {
        launch_cce_add_region(destination.raw_data(),source.raw_data(),source.shape[0],source.shape[1],
                             destination.shape[1],row,col); return;
    }
#endif
    for(int r=0;r<source.shape[0];++r) for(int c=0;c<source.shape[1];++c)
        destination.data()[(r+row)*destination.shape[1]+c+col]+=source.data()[r*source.shape[1]+c];
}
} // namespace nsos::cce
