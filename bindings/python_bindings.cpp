#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector conversion
#include "pantheon_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pantheon, m) {
    m.doc() = "Pantheon Engine Python Bindings via KernelOpen";

    py::class_<pantheon::PantheonEngine>(m, "Engine")
        .def(py::init<>())
        .def("initialize", &pantheon::PantheonEngine::initialize)
        .def("shutdown", &pantheon::PantheonEngine::shutdown)
        .def("submit_dummy_task", &pantheon::PantheonEngine::submit_dummy_task)
        .def("get_throughput", &pantheon::PantheonEngine::get_throughput)
        .def("get_status", &pantheon::PantheonEngine::get_status)

        // Part 1
        .def("compute_contrastive_loss", &pantheon::PantheonEngine::compute_contrastive_loss)
        .def("compute_relational_loss", &pantheon::PantheonEngine::compute_relational_loss)
        .def("compute_msdcrd_loss", &pantheon::PantheonEngine::compute_msdcrd_loss)
        .def("compute_cam_loss", &pantheon::PantheonEngine::compute_cam_loss)
        .def("compute_ntce_loss", &pantheon::PantheonEngine::compute_ntce_loss)
        .def("compute_optimal_transport_loss", &pantheon::PantheonEngine::compute_optimal_transport_loss)

        // Part 2
        .def("compute_chunk_wise_loss", &pantheon::PantheonEngine::compute_chunk_wise_loss)
        .def("adjust_granularity", &pantheon::PantheonEngine::adjust_granularity)
        .def("compute_symbolic_loss", &pantheon::PantheonEngine::compute_symbolic_loss)

        // Part 3
        .def("compute_gradient_loss", &pantheon::PantheonEngine::compute_gradient_loss)
        .def("compute_ib_loss", &pantheon::PantheonEngine::compute_ib_loss)
        .def("compute_topology_loss", &pantheon::PantheonEngine::compute_topology_loss)
        .def("compute_ode_adjoint", &pantheon::PantheonEngine::compute_ode_adjoint)

        // Part 4
        .def("update_meta_policy", &pantheon::PantheonEngine::update_meta_policy)
        .def("compute_causal_loss", &pantheon::PantheonEngine::compute_causal_loss)
        .def("compute_quantum_loss", &pantheon::PantheonEngine::compute_quantum_loss)
        .def("compute_spike_loss", &pantheon::PantheonEngine::compute_spike_loss)

        // Part 5-8
        .def("compute_photonic_loss", &pantheon::PantheonEngine::compute_photonic_loss)
        .def("inject_memristive_noise", &pantheon::PantheonEngine::inject_memristive_noise)
        .def("compute_functorial_loss", &pantheon::PantheonEngine::compute_functorial_loss)
        .def("compute_tom_loss", &pantheon::PantheonEngine::compute_tom_loss)
        .def("compute_nash_regret", &pantheon::PantheonEngine::compute_nash_regret)
        .def("update_swarm", &pantheon::PantheonEngine::update_swarm)
        .def("compute_pate_aggregation", &pantheon::PantheonEngine::compute_pate_aggregation);
}
