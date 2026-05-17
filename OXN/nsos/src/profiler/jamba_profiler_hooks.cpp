// jamba_profiler_hooks.cpp — concrete callbacks the profiler tool
// installs into the JambaModel forward path.  Lives in the
// `nsos_profiler` static library, which is NOT linked into production
// nsos_core builds.
//
// install_hooks() flips two globals (declared as nullptr in jamba.cpp)
// to point at these functions.  The model's forward loop then calls
// them once per layer when profiler_ is non-null.

#include "../../include/profiler/jamba_profiler_hooks.h"
#include "../../include/profiler/inference_profiler.h"

#include <cstdint>

extern "C" {

// Defined as nullptr in jamba.cpp; this TU does NOT redefine them.
extern nsos_profiler_layer_event_fn g_nsos_profiler_begin_layer;
extern nsos_profiler_layer_end_fn   g_nsos_profiler_end_layer;

namespace {

// Per-layer string names.  We pre-build pointers to short literal
// strings to avoid per-call snprintf in the hot path.  The model
// passes layer_idx; we index into this table.  Up to 64 layers
// supported by the prebuilt table; beyond that we fall back to
// the literal "layer_N" generated lazily (the lazy path allocates
// but only on the first encounter, then caches).
constexpr int kPrebuiltLayerCount = 64;
const char* layer_name_table[kPrebuiltLayerCount] = {
    "layer_00", "layer_01", "layer_02", "layer_03", "layer_04",
    "layer_05", "layer_06", "layer_07", "layer_08", "layer_09",
    "layer_10", "layer_11", "layer_12", "layer_13", "layer_14",
    "layer_15", "layer_16", "layer_17", "layer_18", "layer_19",
    "layer_20", "layer_21", "layer_22", "layer_23", "layer_24",
    "layer_25", "layer_26", "layer_27", "layer_28", "layer_29",
    "layer_30", "layer_31", "layer_32", "layer_33", "layer_34",
    "layer_35", "layer_36", "layer_37", "layer_38", "layer_39",
    "layer_40", "layer_41", "layer_42", "layer_43", "layer_44",
    "layer_45", "layer_46", "layer_47", "layer_48", "layer_49",
    "layer_50", "layer_51", "layer_52", "layer_53", "layer_54",
    "layer_55", "layer_56", "layer_57", "layer_58", "layer_59",
    "layer_60", "layer_61", "layer_62", "layer_63"
};
const char* layer_name(int idx) {
    if (idx >= 0 && idx < kPrebuiltLayerCount) return layer_name_table[idx];
    return "layer_overflow";  // very rare; reports under one bucket
}

void on_layer_begin(void* profiler_ptr, int layer_idx) {
    auto* p = reinterpret_cast<nsos::profiler::InferenceProfiler*>(profiler_ptr);
    if (!p) return;
    p->begin_event(nsos::profiler::EventKind::LAYER, layer_name(layer_idx),
                    layer_idx);
}

void on_layer_end(void* profiler_ptr) {
    auto* p = reinterpret_cast<nsos::profiler::InferenceProfiler*>(profiler_ptr);
    if (!p) return;
    p->end_event();
}

}  // namespace

void nsos_profiler_install_hooks() {
    g_nsos_profiler_begin_layer = &on_layer_begin;
    g_nsos_profiler_end_layer   = &on_layer_end;
}

void nsos_profiler_uninstall_hooks() {
    g_nsos_profiler_begin_layer = nullptr;
    g_nsos_profiler_end_layer   = nullptr;
}

}  // extern "C"
