#pragma once
#include <cstddef>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <stdexcept>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NSOS_M3_SHAPE_HD __host__ __device__
#else
#define NSOS_M3_SHAPE_HD
#endif
namespace nsos::mamba3_block {
inline constexpr const char* identity="mamba3_dense_fp32_siso_mimo_n128_v1";
inline constexpr const char* upstream="e9594ce1c732d97440f0332fdc43170a2294dbfa";
// Canonical upstream projection order z,x,B,C,dt,A,trap,angles.
// MIMO rotates coordinates i and i+N/2; SISO rotates 2*i and 2*i+1.
struct Shape {
    int batch=0,sequence=0,model=0,heads=0,groups=0,head_dim=0,state_dim=128,rank=1,rotary_pairs=32;
    bool mimo=false,out_norm=false;
    double norm_eps=1e-5,a_floor=1e-4;
    NSOS_M3_SHAPE_HD int inner() const {return heads*head_dim;}
    NSOS_M3_SHAPE_HD int bc() const {return rank*groups*state_dim;}
    NSOS_M3_SHAPE_HD int width() const {return 2*inner()+2*bc()+3*heads+rotary_pairs;}
};
// Core weights are packed in this order. Every canonical Parameter is a
// non-overlapping owning storage_view, never a duplicate optimizer parameter.
struct Layout {
    std::size_t bnorm=0,cnorm=0,dt=0,bbias=0,cbias=0,d=0,x=0,z=0,o=0,norm=0,total=0;
    NSOS_M3_SHAPE_HD explicit Layout(const Shape& s) {
        cnorm=s.state_dim;dt=cnorm+s.state_dim;bbias=dt+s.heads;
        cbias=bbias+std::size_t(s.heads)*s.rank*s.state_dim;d=cbias+std::size_t(s.heads)*s.rank*s.state_dim;
        x=d+s.heads;z=x+(s.mimo?std::size_t(s.heads)*s.rank*s.head_dim:0);
        o=z+(s.mimo?std::size_t(s.heads)*s.rank*s.head_dim:0);
        norm=o+(s.mimo?std::size_t(s.heads)*s.rank*s.head_dim:0);total=norm+(s.out_norm?s.inner():0);
    }
};
inline bool eligible(const Shape& s) {
    if(s.batch<=0||s.batch>65535||s.sequence<=0||s.sequence>16777216||s.model<=0||s.model>65536||
       s.heads<=0||s.heads>1024||s.groups<=0||s.groups>s.heads||s.heads%s.groups||
       s.head_dim<=0||s.head_dim>128||s.state_dim<=0||s.state_dim>128||s.state_dim%2||
       s.rank<=0||s.rank>8||(!s.mimo&&s.rank!=1)||s.rotary_pairs<=0||s.rotary_pairs>s.state_dim/2||
       !std::isfinite(s.norm_eps)||s.norm_eps<1e-12||s.norm_eps>1||!std::isfinite(s.a_floor)||s.a_floor<=0||s.a_floor>64) return false;
    // Tensor currently has INT_MAX element capacity; guard every dense history,
    // scratch, core partial and projection allocation before constructing it.
    const auto lim=std::size_t(std::numeric_limits<int>::max()),bh=std::size_t(s.batch)*s.heads,t=std::size_t(s.batch)*s.sequence;
    return t<=lim/s.width()&&t<=lim/s.model&&t<=lim/s.inner()&&
        bh<=lim/(std::size_t(s.sequence+1)*s.head_dim*s.state_dim)&&
        bh<=lim/(std::size_t(s.sequence)*s.rank*s.state_dim)&&
        bh<=lim/(std::size_t(s.sequence)*s.rank*s.head_dim)&&
        std::size_t(s.batch)<=lim/Layout(s).total&&std::size_t(s.width())<=lim/s.model;
}
inline std::size_t state_size(const Shape& s) {return std::size_t(s.batch)*s.heads*(s.rotary_pairs+s.head_dim*s.state_dim+s.rank*s.state_dim+s.head_dim);}
inline std::size_t history_size(const Shape& s) {return std::size_t(s.batch)*s.heads*(s.sequence+1)*s.head_dim*s.state_dim;}
inline std::size_t rotation_size(const Shape& s) {return std::size_t(s.batch)*s.heads*s.sequence*s.rank*s.state_dim;}
inline std::size_t phase_size(const Shape& s) {return std::size_t(s.batch)*s.heads*s.sequence*s.rotary_pairs;}
inline std::size_t readout_size(const Shape& s) {return std::size_t(s.batch)*s.heads*s.sequence*s.rank*s.head_dim;}
inline std::size_t scratch_per_head(const Shape& s) {return std::size_t(s.head_dim)*s.state_dim+3*s.rank*s.state_dim+2*s.rank*s.head_dim+s.rotary_pairs;}
template<class T> struct State {T *phase=nullptr,*ssm=nullptr,*k=nullptr,*v=nullptr;};
enum class GpuProvider { DenseReference, ParallelFp32, FlashFp32, FlashFp32ReplayLdsV2, FlashFp32HierarchicalV1 };
inline bool is_flash_provider(GpuProvider p) {return p==GpuProvider::FlashFp32||p==GpuProvider::FlashFp32ReplayLdsV2||p==GpuProvider::FlashFp32HierarchicalV1;}
inline bool is_hierarchical_provider(GpuProvider p) {return p==GpuProvider::FlashFp32HierarchicalV1;}
inline bool is_replay_lds_provider(GpuProvider p) {return p==GpuProvider::FlashFp32ReplayLdsV2||is_hierarchical_provider(p);}
inline GpuProvider gpu_provider_from_environment() {
    const char* value=std::getenv("NSOS_MAMBA3_GPU_PROVIDER");
    const std::string mode=value?value:"dense_reference";
    if(mode=="dense_reference") return GpuProvider::DenseReference;
    if(mode=="parallel_fp32_v1") return GpuProvider::ParallelFp32;
    if(mode=="flash_fp32_v1") return GpuProvider::FlashFp32;
    if(mode=="flash_fp32_hierarchical_v1") return GpuProvider::FlashFp32HierarchicalV1;
    if(mode=="flash_fp32_replay_lds_v2") return GpuProvider::FlashFp32ReplayLdsV2;
    throw std::invalid_argument("Unknown NSOS_MAMBA3_GPU_PROVIDER: "+mode);
}
inline constexpr int parallel_tile=32;
inline std::size_t checkpoint_history_size(const Shape& s) {
    return std::size_t(s.batch)*s.heads*((s.sequence+parallel_tile-1)/parallel_tile+1)*s.head_dim*s.state_dim;
}
inline constexpr int coefficient_fields=7;
inline std::size_t coefficient_size(const Shape& s) {return std::size_t(coefficient_fields)*s.batch*s.heads*s.sequence;}
inline Layout token_layout(Shape s) {s.heads=1;s.groups=1;return Layout(s);}
inline bool parallel_eligible(const Shape& s) {
    const auto count=std::size_t(s.batch)*s.heads*s.sequence,lim=std::size_t(std::numeric_limits<int>::max());
    return eligible(s)&&count<=lim/coefficient_fields&&count<=lim/token_layout(s).total&&count<=lim/(2*s.rank*s.state_dim)&&count<=lim/(s.rank*s.head_dim)&&count<=lim/s.rotary_pairs;
}
// Arity32 inclusive summary tree: Q + ceil(Q/32) + ... (last length <=32) nodes per PN cell.
inline std::size_t hierarchy_nodes_per_cell(const Shape& s) {
    std::size_t q=(s.sequence+parallel_tile-1)/parallel_tile,total=q;
    while(q>parallel_tile) {q=(q+parallel_tile-1)/parallel_tile;total+=q;}return total;
}
inline std::size_t hierarchy_elements(const Shape& s) {return std::size_t(s.batch)*s.heads*s.head_dim*s.state_dim*hierarchy_nodes_per_cell(s);}
inline bool hierarchical_eligible(const Shape& s) {return parallel_eligible(s)&&hierarchy_elements(s)<=std::size_t(std::numeric_limits<int>::max());}
template<class T> struct Trace {T *history=nullptr,*q=nullptr,*k=nullptr,*phase=nullptr,*readout=nullptr;bool parallel=false,checkpoints=false;T* coefficients=nullptr;bool replay_lds=false;bool hierarchical=false;float* hierarchy_a=nullptr;float* hierarchy_b=nullptr;};
struct BackwardWorkspace {
    float *gy=nullptr,*token_parameters=nullptr,*bc=nullptr,*phase=nullptr,*reverse=nullptr;
    float *dz=nullptr,*dx=nullptr,*ddt=nullptr,*da=nullptr,*dtrap=nullptr;
};
// Numeric routines are shared by the FP64 oracle, explicit CPU baseline, and
// GPU correctness baseline. Independent numerical/quadratic tests are required;
// agreement of these three implementations alone is not independent evidence.
bool gpu_forward(Shape s,const float* projection,const float* core,const int* valid,
    State<const float> initial,State<float> final,Trace<float> trace,float* y,float* status);
bool gpu_backward(Shape s,const float* projection,const float* core,const int* valid,
    State<const float> initial,Trace<const float> trace,const float* dy,State<const float> final_seed,
    float* dprojection,float* partial,State<float> dinitial,float* scratch,float* status,BackwardWorkspace workspace={});
bool gpu_reduce(Shape s,const float* partial,float* gradient,const float* status);
// Mask copies never read padded input/adjoint storage. Gate protects state
// publication after the full projected output has been checked for finiteness.
bool gpu_mask(int batch,int sequence,int width,const int* valid,const float* input,float* output);
bool gpu_check(int batch,int sequence,int width,const int* valid,const float* values,float* status);
bool gpu_gate(int batch,std::size_t elements_per_batch,const float* status,float* values,bool whole_op=false);
bool gpu_parameter_check(int batch,std::size_t elements,const float* values,float* status);
bool gpu_parameter_gate(int batch,std::size_t elements,const float* status,float* values);
}

#undef NSOS_M3_SHAPE_HD
