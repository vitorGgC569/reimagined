#include "lean_integration.h"
#include "monitor.h"
#include "rierass_core.h"
#include "cuda/bitnet_math.cuh"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
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
