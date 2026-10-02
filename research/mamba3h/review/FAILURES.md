# Preserved failures and censoring

2026-10-01 native smoke attempts 1/2 failed the initial-phase directional finite
difference: analytic -0.013583145, central difference 8.305319 with step .002.
The objective included a linear final wrapped-phase adjoint. Native phase is
reduced modulo 2*pi; this probe crossed a representation discontinuity. Reducing
the step to .0002 yielded -0.01349681, consistent with the derivative. This is
not evidence of a native smooth-branch VJP bug. Corrected fixture shifts initial
phase by +2 radians before any learned trial, preserving original tolerances.
The original failed attempt2 traceback is retained in native-attempt2.log.

Review also caught an unexecuted parameter-perturbation implementation error:
Tensor.numpy() returns a cloned snapshot, not a mutable native view. The runner
Attempt 3 also rejected buffer mutation: the binding declares buffer protocol
but provides no buffer exporter. Private parameter finite differences are
therefore unattempted through this Python API. Parameter VJP is instead checked
against the pinned upstream Torch reference for eight cases. No production
parameters or files were mutated; tolerances were unchanged.

The native provider rejects physical zero-token input `[2,0,8]` with
`Mamba3 unsupported/overflowing configuration` (native-empty-attempt.log).
Zero valid prefixes on a nonempty padded tensor passed and retain initial
state. Physical empty calls are an unsupported API boundary, not a passing
equivalence case. No production API change was made.

All 30 pilot-v1 factorial jobs completed; zero failed/censored/unattempted
within the declared grid. Every five-seed gain interval versus M0 includes
zero. This is a negative result for demonstrated advantage under this schedule,
not a universal rejection of the architecture. Whole-grid untouched axes are
listed in pilot-v1.json. Algebra rank4 recompression failure is preserved in
the completed Lyra handoff: held-out D64 relative norm error ~1.24.

Final combined peer test runner attempt1 failed during module import: two
independently configured Torch test modules tried to re-set interop threads in
one interpreter. No test failed. The corrected runner isolates each peer suite
in its own process with preimport thread caps, and preserves each result/log.
The original traceback is retained in final-peer-tests-attempt1.log.
