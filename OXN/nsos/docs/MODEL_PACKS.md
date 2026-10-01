# NSOS Model Packs

Model packs are the supported portable format for saving and reloading NSOS
models.

## Contents

A pack directory contains:

- `manifest.nsos`
- immutable generation directories under `generations/`
- within the generation selected by the manifest: `config.nsos`,
  `tokenizer.nsos`, `model.nsos.bin`, and `edge_linear.nsos`

The manifest currently uses:

- `format=nsos-pack-v2`
- `version=2`
- `quantization_ready=0|1`
- mandatory lowercase SHA-256 digests for every child artifact

`quantization_ready=1` is emitted only while the Trainer and model are held in
one transaction, progressive QAT is at or past its effective start step, at
least one deployable `BitLinear` exists, and every non-sensitive deployable
linear is actually on its quantized training path. Merely enabling QAT in the
configuration or completing an FP32 warm-up step is insufficient. This bit and
the exported weights therefore describe the same optimizer generation.

## Security Rules

Pack loading rejects:

- missing required files
- unsupported format/version
- absolute child paths
- `..` path traversal
- files over configured size limits
- missing, malformed, or mismatched SHA-256 digests
- malformed tokenizer packs
- non-contiguous tokenizer ids or BPE merge ranks

Pack children are written into a new, immutable, generation-qualified
directory and flushed to durable storage. The manifest is published last
through one atomic replacement. An interrupted replacement therefore leaves
the preceding manifest and all of its children intact; it cannot corrupt a
previously valid pack by replacing only a subset of stable child filenames.
Unreferenced generation directories may be retained after a process crash or
concurrent publisher and can be garbage-collected only while no pack writer is
active.

Full model checkpoints use a mandatory SHA-256 payload trailer. A new Trainer
sidecar may be written only against a current v4 model checkpoint whose exact
payload equals the live model. A fail-stop optimizer recovery must load that
model checkpoint first and its matching sidecar second; sidecar-only recovery
is rejected. Legacy v1-v3 checkpoints remain readable, but must be loaded and
re-saved before anchoring a new sidecar.

Changing `optimizer_state_bits` between 32 and 4 is canonicalized at the next
optimizer preflight. Sidecar export rejects a mixed or stale representation;
it never guesses which moment state is authoritative.

## Release Expectations

Every release must prove:

- pack save succeeds
- pack reload succeeds
- reload metrics mark `loaded_from_pack`
- tampered weights are rejected
- a missing SHA-256 is rejected (there is no weak-checksum fallback)
- a failed replacement preserves the preceding loadable generation
- manifest traversal is rejected
- pre-QAT exports carry `quantization_ready=0`
- active-QAT exports carry `quantization_ready=1`
- a negative Adam second moment is rejected before any parameter mutation
- an ambiguous mid-commit failure poisons training and export until exact
  model-then-sidecar recovery
- generated output is deterministic across save/reload under the benchmark gate
