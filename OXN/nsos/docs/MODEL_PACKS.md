# NSOS Model Packs

Model packs are the supported portable format for saving and reloading NSOS
models.

## Contents

A pack directory contains:

- `manifest.nsos`
- `config.nsos`
- `tokenizer.nsos`
- `model.nsos.bin`
- `edge_linear.nsos`

The manifest currently uses:

- `format=nsos-pack-v2`
- `version=2`
- FNV checksums for compatibility
- SHA-256 digests for integrity

## Security Rules

Pack loading rejects:

- missing required files
- unsupported format/version
- absolute child paths
- `..` path traversal
- files over configured size limits
- digest/checksum mismatches
- malformed tokenizer packs

## Release Expectations

Every release must prove:

- pack save succeeds
- pack reload succeeds
- reload metrics mark `loaded_from_pack`
- tampered weights are rejected
- manifest traversal is rejected
- generated output is deterministic across save/reload under the benchmark gate
