// profiler_bindings.cpp — pybind11 module `nsos_profiler_ext`.
//
// Exposes the InferenceProfiler + CacheLatencyReport to Python so
// the heatmap_profiler.py driver can:
//   1. Construct a profiler
//   2. attach_profiler(model)
//   3. Run inference
//   4. Drain to JSON / in-memory summary
//   5. Run the cache probe
//
// Built as a separate Python module (nsos_profiler_ext.so) so
// production users who only `import nsos_ext` never even see this
// code on disk.

#include "../../include/profiler/inference_profiler.h"
#include "../../include/profiler/cache_probe.h"
#include "../../include/profiler/jamba_profiler_hooks.h"
#include "../../include/jamba.h"
#include "../../include/nsos_sdk.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(nsos_profiler_ext, m) {
    m.doc() = "NSOS standalone inference profiler.\n"
              "OPT-IN — never linked into production training builds.\n"
              "Captures per-layer cycle counts via RDTSC + a memory\n"
              "hierarchy probe so the architectural analysis report\n"
              "can suggest concrete data-structure changes.";

    // ── EventKind ────────────────────────────────────────────────────
    py::enum_<nsos::profiler::EventKind>(m, "EventKind")
        .value("PHASE",  nsos::profiler::EventKind::PHASE)
        .value("LAYER",  nsos::profiler::EventKind::LAYER)
        .value("OP",     nsos::profiler::EventKind::OP)
        .value("KERNEL", nsos::profiler::EventKind::KERNEL)
        .export_values();

    // ── LayerSummary / OpSummary / ProfilerSummary ──────────────────
    py::class_<nsos::profiler::LayerSummary>(m, "LayerSummary")
        .def_readonly("layer_idx",       &nsos::profiler::LayerSummary::layer_idx)
        .def_readonly("total_cycles",    &nsos::profiler::LayerSummary::total_cycles)
        .def_readonly("call_count",      &nsos::profiler::LayerSummary::call_count)
        .def_readonly("min_cycles",      &nsos::profiler::LayerSummary::min_cycles)
        .def_readonly("max_cycles",      &nsos::profiler::LayerSummary::max_cycles)
        .def_readonly("total_bytes_in",  &nsos::profiler::LayerSummary::total_bytes_in)
        .def_readonly("total_bytes_out", &nsos::profiler::LayerSummary::total_bytes_out)
        .def_readonly("total_flops",     &nsos::profiler::LayerSummary::total_flops);

    py::class_<nsos::profiler::OpSummary>(m, "OpSummary")
        .def_readonly("name",         &nsos::profiler::OpSummary::name)
        .def_readonly("total_cycles", &nsos::profiler::OpSummary::total_cycles)
        .def_readonly("call_count",   &nsos::profiler::OpSummary::call_count)
        .def_readonly("min_cycles",   &nsos::profiler::OpSummary::min_cycles)
        .def_readonly("max_cycles",   &nsos::profiler::OpSummary::max_cycles);

    py::class_<nsos::profiler::ProfilerSummary>(m, "ProfilerSummary")
        .def_readonly("total_events",          &nsos::profiler::ProfilerSummary::total_events)
        .def_readonly("total_cycles_observed", &nsos::profiler::ProfilerSummary::total_cycles_observed)
        .def_readonly("cycles_per_ns",         &nsos::profiler::ProfilerSummary::cycles_per_ns)
        .def_readonly("wall_seconds",          &nsos::profiler::ProfilerSummary::wall_seconds)
        .def_readonly("layers",                &nsos::profiler::ProfilerSummary::layers)
        .def_readonly("ops",                   &nsos::profiler::ProfilerSummary::ops);

    // ── InferenceProfiler ────────────────────────────────────────────
    py::class_<nsos::profiler::InferenceProfiler>(m, "InferenceProfiler")
        .def(py::init<size_t>(), py::arg("ring_capacity") = 65536)
        .def("reset",                  &nsos::profiler::InferenceProfiler::reset)
        .def("recalibrate",            &nsos::profiler::InferenceProfiler::recalibrate,
             py::arg("sample_ms") = 50)
        .def("drain_to_json",          &nsos::profiler::InferenceProfiler::drain_to_json,
             py::arg("path"),
             "Write events + summaries to a JSON file.  Atomic rename.")
        .def("drain_summary",          &nsos::profiler::InferenceProfiler::drain_summary,
             "Return in-memory ProfilerSummary (per-layer + per-op aggregates).")
        .def("event_count_observed",   &nsos::profiler::InferenceProfiler::event_count_observed)
        .def("event_count_dropped",    &nsos::profiler::InferenceProfiler::event_count_dropped)
        // Attach / detach helpers.  attach takes any object with a
        // .ptr() attribute (the InferenceProfiler itself) but we
        // also expose them as standalone functions for convenience.
        ;

    // ── Cache probe ─────────────────────────────────────────────────
    py::class_<nsos::profiler::CacheLatencySample>(m, "CacheLatencySample")
        .def_readonly("working_set_bytes",        &nsos::profiler::CacheLatencySample::working_set_bytes)
        .def_readonly("median_cycles_per_access", &nsos::profiler::CacheLatencySample::median_cycles_per_access)
        .def_readonly("p90_cycles_per_access",    &nsos::profiler::CacheLatencySample::p90_cycles_per_access)
        .def_readonly("accesses_measured",        &nsos::profiler::CacheLatencySample::accesses_measured);

    py::class_<nsos::profiler::CacheLatencyReport>(m, "CacheLatencyReport")
        .def_readonly("samples",            &nsos::profiler::CacheLatencyReport::samples)
        .def_readonly("inferred_l1_bytes",  &nsos::profiler::CacheLatencyReport::inferred_l1_bytes)
        .def_readonly("inferred_l2_bytes",  &nsos::profiler::CacheLatencyReport::inferred_l2_bytes)
        .def_readonly("inferred_l3_bytes",  &nsos::profiler::CacheLatencyReport::inferred_l3_bytes)
        .def_readonly("dram_latency_cycles", &nsos::profiler::CacheLatencyReport::dram_latency_cycles);

    m.def("probe_cache_latencies", &nsos::profiler::probe_cache_latencies,
          "Run the pointer-chase memory hierarchy probe.  Takes ~1-2 seconds.");

    // ── Hook installation ───────────────────────────────────────────
    m.def("install_hooks",   &nsos_profiler_install_hooks,
          "Install the profiler callbacks into JambaModel's forward path.  "
          "Call once at the start of a profiling session.");
    m.def("uninstall_hooks", &nsos_profiler_uninstall_hooks,
          "Restore the null callbacks (turn off profiling globally).");

    // attach_profiler / detach_profiler on JambaModel.  Takes a
    // capsule wrapping the InferenceProfiler pointer.  We forward
    // via void* through the JambaModel API which doesn't include
    // the profiler header in production.
    m.def("attach_to_model",
          [](py::object model, nsos::profiler::InferenceProfiler& profiler) {
              auto* m = model.cast<nsos::JambaModel*>();
              m->attach_profiler(static_cast<void*>(&profiler));
          },
          py::arg("model"), py::arg("profiler"),
          "Attach an InferenceProfiler to a JambaModel.  Subsequent "
          "forward passes will emit per-layer events into the profiler's "
          "ring buffer until detach_from_model is called.");
    m.def("detach_from_model",
          [](py::object model) {
              auto* m = model.cast<nsos::JambaModel*>();
              m->attach_profiler(nullptr);
          },
          py::arg("model"));

    // Engine-level convenience: attach/detach via the InferenceEngine
    // wrapper.  Saves the caller from having to expose engine.model
    // separately.
    m.def("attach_to_engine",
          [](py::object engine_obj, nsos::profiler::InferenceProfiler& profiler) {
              auto* engine = engine_obj.cast<nsos::InferenceEngine*>();
              if (!engine || !engine->model) {
                  throw std::runtime_error(
                      "attach_to_engine: engine has no loaded model");
              }
              // Set the opaque data pointer (the profiler instance) AND
              // the function-pointer callbacks on the model instance.
              // Storing function pointers on the instance (not DLL
              // globals) sidesteps the Windows two-pyd boundary issue:
              // the model code in nsos_ext.pyd and the callback code in
              // nsos_profiler_ext.pyd live in different DLLs but share
              // the same process address space, so function-pointer
              // values are uniformly callable.
              engine->model->attach_profiler(static_cast<void*>(&profiler));
              engine->model->set_profiler_callbacks(
                  &nsos_profiler_on_layer_begin,
                  &nsos_profiler_on_layer_end);
          },
          py::arg("engine"), py::arg("profiler"),
          "Attach a profiler to a fully-loaded InferenceEngine.  Installs "
          "both the data pointer (the profiler itself) AND the function-"
          "pointer callbacks on the model instance.  No globals involved, "
          "so works across the nsos_ext.pyd / nsos_profiler_ext.pyd DLL "
          "boundary.");

    m.def("detach_from_engine",
          [](py::object engine_obj) {
              auto* engine = engine_obj.cast<nsos::InferenceEngine*>();
              if (engine && engine->model) {
                  engine->model->attach_profiler(nullptr);
                  engine->model->set_profiler_callbacks(nullptr, nullptr);
              }
          },
          py::arg("engine"));
}
