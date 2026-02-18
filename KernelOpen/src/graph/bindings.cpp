#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sssp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(uhk_graph, m) {
    m.doc() = "Universal Heterogeneous Kernel - Graph Module (CHRASS)";

    py::class_<uhk::graph::SSSP>(m, "SSSP")
        .def_static("solve", [](int n, py::array_t<int> row_ptr, py::array_t<int> col_ind, py::array_t<int> values, int source) {
            uhk::graph::GraphCSR graph;
            graph.n = n;

            // Zero-Copy Check (optional safety)
            // Assuming int32 inputs
            py::buffer_info r_buf = row_ptr.request();
            py::buffer_info c_buf = col_ind.request();
            py::buffer_info v_buf = values.request();

            // Copy to Vector (for safety in this prototype, zero-copy requires span/view in logic)
            // CHRASS logic takes vectors.
            graph.row_ptr.assign((int*)r_buf.ptr, (int*)r_buf.ptr + r_buf.size);
            graph.col_ind.assign((int*)c_buf.ptr, (int*)c_buf.ptr + c_buf.size);
            graph.values.assign((int*)v_buf.ptr, (int*)v_buf.ptr + v_buf.size);

            auto [dist, ops, ms] = uhk::graph::SSSP::solve(graph, source);

            // Return tuple: (dist_array, ops, ms)
            return py::make_tuple(
                py::array(dist.size(), dist.data()),
                ops,
                ms
            );
        }, "Solve SSSP using CHRASS (Radix Heap + AVX2)");
}
