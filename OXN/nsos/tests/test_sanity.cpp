#include "lean_integration.h"
#include "monitor.h"
#include "rierass_core.h"
#include "cuda/bitnet_math.cuh"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace nsos;
void check(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
void empty_contract(const Tensor& original) {
    check(original.size==0,"fixture must be empty");
    const auto bytes0=gpu_transfer_stats();
    for(int repetition=0;repetition<8;++repetition) {
        auto cloned=original.clone();
        check(cloned.size==0 && cloned.shape.dims==original.shape.dims && cloned.shape.strides==original.shape.strides,"clone empty metadata");
        check(cloned.get_device()==original.get_device() && cloned.raw_data()==nullptr && !cloned.grad,"clone empty allocates/aliases data or grad");
        for(auto dev:{Device::CPU,Device::GPU}) {
            auto moved=cloned.to(dev);
            check(moved.size==0 && moved.shape.dims==original.shape.dims && moved.shape.strides==original.shape.strides,"to empty metadata");
            check(moved.get_device()==dev && moved.raw_data()==nullptr,"empty device conversion allocates");
            auto host=moved.cpu().clone();
            check(host.size==0 && host.get_device()==Device::CPU && host.raw_data()==nullptr,"empty cpu round trip");
        }
    }
    const auto bytes1=gpu_transfer_stats();
    check(bytes0.h2d_calls==bytes1.h2d_calls && bytes0.d2h_calls==bytes1.d2h_calls &&
        bytes0.d2d_calls==bytes1.d2d_calls && bytes0.h2h_calls==bytes1.h2h_calls &&
        bytes0.device_synchronizations==bytes1.device_synchronizations && bytes0.stream_synchronizations==bytes1.stream_synchronizations,
        "empty clone/to caused transfer or GPU synchronization");
}
void empty_tensor_sanity() {
    empty_contract(Tensor());
    for(auto shape:{std::vector<int>{0},{2,0,3},{0,4},{3,2,0},{0,0}})
        for(auto dev:{Device::CPU,Device::GPU}) empty_contract(Tensor(shape,dev));
    Tensor parent({4},Device::CPU,3.f);
    auto end_view=parent.storage_view(4,{0});empty_contract(end_view);
    auto sentinel=Tensor();sentinel.grad=std::make_shared<Tensor>(std::vector<int>{1});empty_contract(sentinel);
    Tensor scalar(std::vector<int>{},Device::CPU,1.25f);
    auto cloned=scalar.clone();
    check(cloned.size==1 && cloned.shape.dims.empty() && cloned.raw_data()!=scalar.raw_data() && cloned.data()[0]==1.25f,"real scalar clone changed");
    cloned.data()[0]=-2.f;check(scalar.data()[0]==1.25f,"scalar clone aliases original");
    check(scalar.to(Device::CPU).raw_data()==scalar.raw_data(),"same-device nonempty alias behavior changed");
    auto from_scalar=Tensor::from_scalar(3.5f),from_copy=from_scalar.clone();
    check(from_copy.size==1 && from_copy.shape==from_scalar.shape && from_copy.data()[0]==3.5f,"from_scalar became empty");
    auto matrix=Tensor::ones({2,3});auto copied=matrix.clone();copied.data()[0]=9;
    check(copied.size==6 && matrix.data()[0]==1,"nonempty clone regressed");
    std::cout<<"PASS empty sentinel/zero extents/views/CPU-GPU metadata without transfer; real scalar and nonempty clone independent\n";
}
}

int main() {
    empty_tensor_sanity();
    const std::vector<int8_t> ternary = {0, 1, -1, 0, -1};
    const std::vector<uint8_t> packed =
        uhk::math::pack_ternary_weights(ternary);
    assert(packed.size() == 2);
    assert(packed[0] == 0x24u);
    assert(packed[1] == 0x02u);

    rierass::IsomorphicBuffer scratchpad(2);
    nsos::Tensor thought_a({1, 2}, nsos::Device::CPU, 0.0f);
    nsos::Tensor thought_b({1, 2}, nsos::Device::CPU, 0.0f);
    thought_a.data()[0] = 1.0f;
    thought_a.data()[1] = 2.0f;
    thought_b.data()[0] = 3.0f;
    thought_b.data()[1] = 4.0f;
    scratchpad.write(thought_a);
    scratchpad.write(thought_b);
    const nsos::Tensor sequence = scratchpad.get_sequence();
    assert(sequence.shape.dims == std::vector<int>({2, 2}));
    assert(sequence.data()[0] == 1.0f && sequence.data()[3] == 4.0f);
    const nsos::Tensor linear = scratchpad.read_linear();
    assert(linear.shape.dims == std::vector<int>({1, 4}));
    bool overflow_rejected = false;
    try {
        scratchpad.write(thought_a);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    assert(overflow_rejected);

    nsos::Monitor& monitor = nsos::Monitor::instance();
    monitor.enable();
    monitor.set_verbose(false);

    const nsos::Tensor cpu =
        nsos::Tensor::from_scalar(1.0f, nsos::Device::CPU);
    monitor.check(cpu, "sanity.cpu");
    nsos::Tensor nonfinite =
        nsos::Tensor::from_scalar(0.0f, nsos::Device::CPU);
    nonfinite.data()[0] = std::numeric_limits<float>::quiet_NaN();
    bool nonfinite_rejected = false;
    try {
        monitor.check(nonfinite, "sanity.nonfinite");
    } catch (const std::runtime_error&) {
        nonfinite_rejected = true;
    }
    assert(nonfinite_rejected);
#ifdef USE_CUDA
    const nsos::Tensor gpu =
        nsos::Tensor::from_scalar(1.0f, nsos::Device::GPU);
    monitor.check(gpu, "sanity.gpu");
#endif
    monitor.disable();

    const nsos::LeanVerifier verifier;
    assert(!verifier.available());
    bool rejected_without_provider = false;
    try {
        static_cast<void>(verifier.verify("1+1=2"));
    } catch (const std::runtime_error&) {
        rejected_without_provider = true;
    }
    assert(rejected_without_provider);

    std::cout << "Sanity check PASSED." << std::endl;
    return 0;
}
