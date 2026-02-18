#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/LinearModel.hpp"
#include "../include/Hilbert.hpp"
#include "../include/BitPacking.hpp"
#include "../../../KernelOpen/include/aion/SpectralLayout.hpp" // New Include

namespace py = pybind11;

PYBIND11_MODULE(aion_core, m) {
    m.doc() = "AION Core C++ Implementation optimized with AVX/AVX2";

    // Bind LinearModel
    py::class_<Aion::LinearModel>(m, "LinearModel")
        .def(py::init<>())
        .def("train", &Aion::LinearModel::train, "Train the linear model")
        .def("predict", &Aion::LinearModel::predict, "Predict value for a key")
        .def("predict_batch", [](Aion::LinearModel& self, py::array_t<double> keys) {
            py::buffer_info buf = keys.request();
            if (buf.ndim != 1) throw std::runtime_error("Number of dimensions must be one");
            size_t n = buf.size;
            auto result = py::array_t<double>(n);
            py::buffer_info res_buf = result.request();
            double* ptr_in = static_cast<double*>(buf.ptr);
            double* ptr_out = static_cast<double*>(res_buf.ptr);
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

    // Bind BitPacker
    py::class_<Aion::BitPacker>(m, "BitPacker")
        .def_static("pack", [](py::array_t<uint32_t> input, int bits) {
            py::buffer_info buf = input.request();
            if (buf.ndim != 1) throw std::runtime_error("Number of dimensions must be one");
            size_t n = buf.size;
            uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);
            size_t out_size = (n * bits + 63) / 64;
            auto result = py::array_t<uint64_t>(out_size);
            py::buffer_info res_buf = result.request();
            uint64_t* out_ptr = static_cast<uint64_t*>(res_buf.ptr);
            Aion::BitPacker::pack_avx512(ptr, out_ptr, n, bits);
            return result;
        }, "Pack 32-bit integers into N-bit stream");

    // --- NEW: Bind SpectralLayout ---
    py::class_<Aion::SpectralLayout>(m, "SpectralLayout")
        .def_static("compute_layout", [](int n, py::array_t<int> row_ptr, py::array_t<int> col_ind, int source_hint) {
            // Convert Numpy to C++ Struct
            Aion::SimpleCSR graph;
            graph.n = n;

            py::buffer_info r_buf = row_ptr.request();
            py::buffer_info c_buf = col_ind.request();

            graph.row_ptr.assign((int*)r_buf.ptr, (int*)r_buf.ptr + r_buf.size);
            graph.col_ind.assign((int*)c_buf.ptr, (int*)c_buf.ptr + c_buf.size);

            // Call Core
            std::vector<int> mapping = Aion::SpectralLayout::compute_layout(graph, source_hint);

            // Return Numpy
            return py::array(mapping.size(), mapping.data());
        }, "Compute Spectral Layout reordering map");
}
