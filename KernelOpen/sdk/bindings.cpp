#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../src/host/uhk_runtime.h"
#include "../src/common/uhk_types.h"

#include "../src/ghost/holographic_memory.h"
#include "../src/ghost/context_predictor.h"
#include "../src/bci/spike_encoder.h"
#include "../src/photonic/interference_engine.h"

namespace py = pybind11;

// Wrapper function to handle numpy arrays and pointers
void submit_bitnet_gemm_wrapper(
    uhk::runtime::UniversalKernelRuntime& self,
    int M, int N, int K,
    py::array_t<uint8_t> A, // Packed Weights
    py::array_t<int8_t> B,  // Activations
    py::array_t<float> C    // Output
) {
    // Request buffer info
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();

    // Check dimensions (simplified checks)
    if (bufA.ndim != 1 || bufB.ndim != 1 || bufC.ndim != 1) {
        throw std::runtime_error("Buffers must be 1D (flat) for this raw interface demo.");
    }

    // Call internal C++ method
    self.submit_bitnet_gemm(
        M, N, K,
        bufA.ptr,
        bufB.ptr,
        bufC.ptr
    );
}

PYBIND11_MODULE(uhk_core, m) {
    m.doc() = "Universal Heterogeneous Kernel (UHK) Python Bindings";

    // Expose OpType Enum
    py::enum_<OpType>(m, "OpType")
        .value("NO_OP", OpType::NO_OP)
        .value("GEMM_BITNET", OpType::GEMM_BITNET)
        .value("GEMM_SYNTHETIC", OpType::GEMM_SYNTHETIC)
        .value("SHUTDOWN", OpType::SHUTDOWN)
        .export_values();

    // Expose Runtime Class
    py::class_<uhk::runtime::UniversalKernelRuntime>(m, "UniversalKernelRuntime")
        .def(py::init<>())
        .def("shutdown", &uhk::runtime::UniversalKernelRuntime::shutdown)
        .def("submit_bitnet_gemm", &submit_bitnet_gemm_wrapper,
             "Submit a BitNet GEMM task. Args: M, N, K, A (uint8), B (int8), C (float).")
        .def("get_throughput", &uhk::runtime::UniversalKernelRuntime::get_throughput)
        .def("get_latency", &uhk::runtime::UniversalKernelRuntime::get_latency)
        .def("get_active_sms", &uhk::runtime::UniversalKernelRuntime::get_active_sms)
        .def("get_ring_buffer_usage", &uhk::runtime::UniversalKernelRuntime::get_ring_buffer_usage)
        .def("get_tasks_processed", &uhk::runtime::UniversalKernelRuntime::get_tasks_processed);

    // --- Ghost Kernel Bindings ---
    py::class_<uhk::ghost::HolographicMemory>(m, "HolographicMemory")
        .def(py::init<size_t>())
        .def("encode", &uhk::ghost::HolographicMemory::encode)
        .def("store", &uhk::ghost::HolographicMemory::store)
        .def("recall", &uhk::ghost::HolographicMemory::recall);

    py::class_<uhk::ghost::ContextPredictor>(m, "ContextPredictor")
        .def(py::init<>())
        .def("observe", &uhk::ghost::ContextPredictor::observe)
        .def("predict", &uhk::ghost::ContextPredictor::predict);

    // --- BCI Bindings ---
    py::class_<uhk::bci::SpikeEncoder>(m, "SpikeEncoder")
        .def(py::init<float, float>())
        .def("step", &uhk::bci::SpikeEncoder::step)
        .def("encode_signal", &uhk::bci::SpikeEncoder::encode_signal);

    // --- Photonic Bindings ---
    py::class_<uhk::photonic::InterferenceEngine>(m, "InterferenceEngine")
        .def(py::init<int>())
        .def("phase_shift", &uhk::photonic::InterferenceEngine::phase_shift)
        .def("compute", &uhk::photonic::InterferenceEngine::compute);
}
