#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/Tensor.h"
#include "../include/LoRA.h"
#include "../include/FullFinetuning.h"
#include "../include/IA3.h"
#include "../include/DoRA.h"
#include "../include/TurboFusion.h"

namespace py = pybind11;

PYBIND11_MODULE(peft_framework, m) {
    m.doc() = "PEFT Framework in C++";

    // Tensor Binding
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def("get_rows", &Tensor::getRows)
        .def("get_cols", &Tensor::getCols)
        .def("at", [](Tensor& t, int r, int c) { return t.at(r, c); })
        .def("set", [](Tensor& t, int r, int c, float val) { t.at(r, c) = val; })
        .def("transpose", &Tensor::transpose);

    // Full Finetuning Binding
    py::class_<FullFinetuningLayer>(m, "FullFinetuningLayer")
        .def(py::init<int, int>(), py::arg("input_dim"), py::arg("output_dim"))
        .def("forward", &FullFinetuningLayer::forward)
        .def("backward", &FullFinetuningLayer::backward);

    // LoRA Binding
    py::class_<LoRALayer>(m, "LoRALayer")
        .def(py::init<int, int, int>(), py::arg("input_dims"), py::arg("output_dims"), py::arg("rank"))
        .def("set_base_weights", &LoRALayer::setBaseWeights)
        .def("forward", &LoRALayer::forward)
        .def("backward", &LoRALayer::backward);

    // IA3 Binding
    py::class_<IA3Layer>(m, "IA3Layer")
        .def(py::init<int, int>(), py::arg("input_dim"), py::arg("output_dim"))
        .def("set_base_weights", &IA3Layer::setBaseWeights)
        .def("forward", &IA3Layer::forward)
        .def("backward", &IA3Layer::backward);

    // DoRA Binding
    py::class_<DoRALayer>(m, "DoRALayer")
        .def(py::init<int, int, int>(), py::arg("input_dims"), py::arg("output_dims"), py::arg("rank"))
        .def("set_base_weights", &DoRALayer::setBaseWeights)
        .def("forward", &DoRALayer::forward)
        .def("backward", &DoRALayer::backward);

    // TurboFusion Binding
    py::class_<TurboFusionLayer>(m, "TurboFusionLayer")
        .def(py::init<int, int, int>(), py::arg("input_dims"), py::arg("output_dims"), py::arg("rank"))
        .def("set_base_weights", &TurboFusionLayer::setBaseWeights)
        .def("forward", &TurboFusionLayer::forward)
        .def("backward", &TurboFusionLayer::backward);
}
