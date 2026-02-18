#include "../include/chrass_layer.h"
#include "../include/components.h"
#include "../include/embedding.h"
#include "../include/holographic.h"
#include "../include/inspector.h"
#include "../include/jamba.h"
#include "../include/kan.h"
#include "../include/mcts_reasoning.h"
#include "../include/memory_system.h"
#include "../include/monitor.h"
#include "../include/nsos_mpi.h"
#include "../include/nsos_sdk.h"
#include "../include/smart_loader.h"
#include "../include/sprecher_kan.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/trainer.h"
#include "../include/ttt_layer.h"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <iostream>
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace nsos;

PYBIND11_MODULE(nsos_ext, m) {
    m.doc() = "NSOS V1.0 - Industrial Cognitive Kernel";

    // Enums
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    // Configuration
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("d_model", &ModelConfig::d_model)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("use_cuda", &ModelConfig::use_cuda)
        .def_readwrite("mcts_simulations", &ModelConfig::mcts_simulations);

    // Tensor
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int>, Device, float>(), py::arg("shape"), py::arg("device")=Device::CPU, py::arg("fill")=0.0f)
        .def("to", &Tensor::to)
        .def("cpu", &Tensor::cpu)
        .def("numpy", [](Tensor& t) {
            if (t.device == Device::GPU) throw std::runtime_error("Move to CPU first");
            return py::array_t<float>(t.shape.dims, t.data());
        })
        .def("add", &Tensor::add)
        .def("sub", &Tensor::sub)
        .def("mul", py::overload_cast<const Tensor&>(&Tensor::mul, py::const_))
        .def("matmul", &Tensor::matmul)
        .def("sum", &Tensor::sum, py::arg("dim")=-1, py::arg("keepdim")=false)
        .def("rmsnorm", &Tensor::rmsnorm)
        .def("softmax", &Tensor::softmax)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("device")=Device::CPU)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device")=Device::CPU)
        .def_static("random", &Tensor::random, py::arg("shape"), py::arg("device")=Device::CPU);

    // ChrassLayer (The Mind)
    py::class_<ChrassLayer>(m, "ChrassLayer")
        .def(py::init<int, const std::vector<float>&>())
        .def("forward", &ChrassLayer::forward)
        .def("step", &ChrassLayer::step)
        .def("to_dense", &ChrassLayer::to_dense);

    // JambaModel (The Body)
    py::class_<JambaModel>(m, "JambaModel")
        .def(py::init<int, int, int, Device>())
        .def("forward", py::overload_cast<const Tensor&, Context*>(&JambaModel::forward))
        .def("reason", &JambaModel::reason)
        .def("save", &JambaModel::save)
        .def("load", &JambaModel::load)
        .def("to", &JambaModel::to)
        .def("parameters", &JambaModel::parameters);

    // MCTS (System 2)
    py::class_<MCTSReasoning>(m, "MCTSReasoning")
        .def(py::init<const Tensor&, MCTSReasoning::Evaluator, const MCTSConfig&>())
        .def("search", &MCTSReasoning::search)
        .def("get_best_state", &MCTSReasoning::get_best_state);

    // Trainer
    py::class_<Trainer>(m, "Trainer")
        .def(py::init<JambaModel*, float>())
        .def("train_step", &Trainer::train_step);

    // MPI
    py::class_<MpiManager>(m, "MpiManager")
        .def_static("init", [](std::vector<std::string> args) {
            // Convert vector string to char**
            // Simplified for pybind
        })
        .def_static("rank", &MpiManager::rank)
        .def_static("size", &MpiManager::size);
}
