#pragma once
#include "nsos_config.h"
#include "tensor.h"
#include <vector>
#include <iostream>
#include <omp.h>

namespace nsos {

struct TensorIterator {
    float* out_ptr;
    const float* in1_ptr;
    const float* in2_ptr;
    
    std::vector<size_t> out_strides;
    std::vector<size_t> in1_strides;
    std::vector<size_t> in2_strides;
    std::vector<int> shape;
    size_t numel;
    int rank;
    
    TensorIterator(Tensor& out, const Tensor& in1, const Tensor& in2) {
        if (!compute_broadcast_shape(in1.shape, in2.shape, out.shape))
            throw std::runtime_error("Broadcasting Error");
            
        shape = out.shape.dims;
        numel = out.size;
        rank = (int)shape.size();
        
        out_ptr = out.data();
        in1_ptr = in1.data();
        in2_ptr = in2.data();
        
        out_strides.resize(rank);
        in1_strides.resize(rank);
        in2_strides.resize(rank);
        
        int r1 = (int)in1.shape.size();
        int r2 = (int)in2.shape.size();
        
        for (int i = 0; i < rank; ++i) {
            out_strides[i] = out.shape.strides[i];
            
            int dim1_idx = i - (rank - r1);
            if (dim1_idx >= 0 && in1.shape.dims[dim1_idx] > 1) 
                in1_strides[i] = in1.shape.strides[dim1_idx];
            else in1_strides[i] = 0;
            
            int dim2_idx = i - (rank - r2);
            if (dim2_idx >= 0 && in2.shape.dims[dim2_idx] > 1) 
                in2_strides[i] = in2.shape.strides[dim2_idx];
            else in2_strides[i] = 0;
        }
    }
    
    static bool compute_broadcast_shape(const TensorShape& s1, const TensorShape& s2, TensorShape& out) {
        int r1 = (int)s1.size();
        int r2 = (int)s2.size();
        int max_r = std::max(r1, r2);
        out.dims.resize(max_r);
        for (int i = 0; i < max_r; ++i) {
            int d1 = (i < max_r - r1) ? 1 : s1.dims[i - (max_r - r1)];
            int d2 = (i < max_r - r2) ? 1 : s2.dims[i - (max_r - r2)];
            if (d1 != d2 && d1 != 1 && d2 != 1) return false;
            out.dims[i] = std::max(d1, d2);
        }
        out.compute_strides();
        return true;
    }
    
    template <typename Func>
    void parallel_for_each(Func op) {
        if (rank == 1) loop_1d(op);
        else if (rank == 2) loop_2d(op);
        else if (rank == 3) loop_3d(op);
        else loop_nd(op);
    }
    
    template <typename Func>
    void loop_1d(Func op) {
        int dim0 = shape[0];
        #pragma omp parallel for
        for(int i=0; i<dim0; ++i) {
            out_ptr[i*out_strides[0]] = op(in1_ptr[i*in1_strides[0]], in2_ptr[i*in2_strides[0]]);
        }
    }
    
    template <typename Func>
    void loop_2d(Func op) {
        int dim0 = shape[0];
        int dim1 = shape[1];
        #pragma omp parallel for
        for(int i=0; i<dim0; ++i) {
            for(int j=0; j<dim1; ++j) {
                size_t off_out = i*out_strides[0] + j*out_strides[1];
                size_t off_in1 = i*in1_strides[0] + j*in1_strides[1];
                size_t off_in2 = i*in2_strides[0] + j*in2_strides[1];
                out_ptr[off_out] = op(in1_ptr[off_in1], in2_ptr[off_in2]);
            }
        }
    }
    
    template <typename Func>
    void loop_3d(Func op) {
        int dim0 = shape[0];
        int dim1 = shape[1];
        int dim2 = shape[2];
        #pragma omp parallel for
        for(int i=0; i<dim0; ++i) {
            for(int j=0; j<dim1; ++j) {
                size_t base_out = i*out_strides[0] + j*out_strides[1];
                size_t base_in1 = i*in1_strides[0] + j*in1_strides[1];
                size_t base_in2 = i*in2_strides[0] + j*in2_strides[1];
                for(int k=0; k<dim2; ++k) {
                    out_ptr[base_out + k*out_strides[2]] = op(in1_ptr[base_in1 + k*in1_strides[2]], in2_ptr[base_in2 + k*in2_strides[2]]);
                }
            }
        }
    }
    
    template <typename Func>
    void loop_nd(Func op) {
        int outer_dim = shape[0];
        #pragma omp parallel for
        for (int i = 0; i < outer_dim; ++i) {
            recursive_loop(1, out_ptr + i*out_strides[0], in1_ptr + i*in1_strides[0], in2_ptr + i*in2_strides[0], op);
        }
    }
    
    template <typename Func>
    void recursive_loop(int d, float* p_out, const float* p_in1, const float* p_in2, Func op) {
        if (d == rank) { *p_out = op(*p_in1, *p_in2); return; }
        int dim = shape[d];
        for (int i = 0; i < dim; ++i) {
            recursive_loop(d + 1, p_out + i*out_strides[d], p_in1 + i*in1_strides[d], p_in2 + i*in2_strides[d], op);
        }
    }
};

} // namespace nsos
