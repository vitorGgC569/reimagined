#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aion/LinearModel.hpp"
#include "aion/Hilbert.hpp"
#include "aion/BitPacking.hpp"

namespace py = pybind11;

PYBIND11_MODULE(aion_core, m) {
    m.doc() = "AION Core C++ Implementation optimized with AVX/AVX2";

    // Bind LinearModel
    py::class_<Aion::LinearModel>(m, "LinearModel")
        .def(py::init<>())
        .def("train", &Aion::LinearModel::train, "Train the linear model")
        .def("predict", &Aion::LinearModel::predict, "Predict value for a key")
        .def("predict_batch", [](Aion::LinearModel& self, py::array_t<double> keys) {
            // Check input
            py::buffer_info buf = keys.request();
            if (buf.ndim != 1) throw std::runtime_error("Number of dimensions must be one");

            size_t n = buf.size;
            auto result = py::array_t<double>(n);
            py::buffer_info res_buf = result.request();

            double* ptr_in = static_cast<double*>(buf.ptr);
            double* ptr_out = static_cast<double*>(res_buf.ptr);

            // Call AVX2 optimized batch method
            self.predict_batch(ptr_in, ptr_out, n);

            return result;
        }, "Predict values for a batch of keys using AVX2 SIMD");

    // Bind Hilbert
    py::class_<Aion::Hilbert>(m, "Hilbert")
        .def_static("xy2d", &Aion::Hilbert::xy2d, "Convert (x,y) to Hilbert distance d")
        .def_static("d2xy", [](int n, int d) {
            int x, y;
            Aion::Hilbert::d2xy(n, d, &x, &y);
            return std::make_pair(x, y);
        }, "Convert Hilbert distance d to (x,y)");

    // Bind BitPacker (expose wrapper for numpy)
    py::class_<Aion::BitPacker>(m, "BitPacker")
        .def_static("pack", [](py::array_t<uint32_t> input, int bits) {
            py::buffer_info buf = input.request();
            if (buf.ndim != 1) throw std::runtime_error("Number of dimensions must be one");

            size_t n = buf.size;
            uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);

            // Calc output size: ceil(n * bits / 64)
            size_t out_size = (n * bits + 63) / 64;
            auto result = py::array_t<uint64_t>(out_size);
            py::buffer_info res_buf = result.request();
            uint64_t* out_ptr = static_cast<uint64_t*>(res_buf.ptr);

            // Call AVX-512 safe wrapper
            Aion::BitPacker::pack_avx512(ptr, out_ptr, n, bits);

            return result;
        }, "Pack 32-bit integers into N-bit stream");
}
