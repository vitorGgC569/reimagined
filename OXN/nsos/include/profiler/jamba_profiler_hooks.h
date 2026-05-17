// jamba_profiler_hooks.h — C ABI thunks between JambaModel and the
// profiler library.
//
// Architecture:
//   The JambaModel forward path declares two function-pointer globals
//   with default = nullptr.  When attach_profiler() is called by the
//   profiler tool, those globals are populated with concrete callbacks
//   defined in src/profiler/jamba_profiler_hooks.cpp.  When the tool
//   is NOT used (production build), the globals stay null, the
//   if-check in forward() skips, and the profiler library is not
//   linked at all.
//
// Why C ABI:
//   The hooks live in a separate static library that's compiled
//   independently of nsos_core.  Using a C ABI guarantees zero
//   name-mangling surprises across compilers and lets us treat the
//   pointers as plain `void*` in jamba.cpp without including any
//   profiler headers there.

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Function pointer types for the layer-begin / layer-end hooks.
// Both take an opaque `void* profiler` (the InferenceProfiler*
// returned by JambaModel::profiler()) and a layer index.
typedef void (*nsos_profiler_layer_event_fn)(void* profiler, int layer_idx);
typedef void (*nsos_profiler_layer_end_fn)(void* profiler);

// Globals that the model forward path consults.  Default = nullptr
// so production paths skip; the profiler tool initializes them via
// `nsos_profiler_install_hooks()` below.
extern nsos_profiler_layer_event_fn g_nsos_profiler_begin_layer;
extern nsos_profiler_layer_end_fn   g_nsos_profiler_end_layer;

// Install the concrete callbacks.  Called once by the profiler tool
// after it constructs an InferenceProfiler.
void nsos_profiler_install_hooks();
// Uninstall (sets the globals back to nullptr).
void nsos_profiler_uninstall_hooks();

// Direct callbacks — exported so profiler_bindings.cpp can take
// their address and pass them through set_profiler_callbacks on
// the model instance.  This is the cross-DLL-safe path.
void nsos_profiler_on_layer_begin(void* profiler_ptr, int layer_idx);
void nsos_profiler_on_layer_end(void* profiler_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif
