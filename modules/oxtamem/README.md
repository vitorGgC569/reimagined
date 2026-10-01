# OxtaMem

OxtaMem is a local Rust/Python causal memory component used as a companion to
NSOS. For the NSOS MVP it is **not** a public network service.

## MVP Status

- Native Rust engine: beta.
- Durable writes use a checksummed append-only metadata journal with torn-tail
  recovery and amortized snapshot compaction; full metadata is not rewritten
  for every record.
- Python SDK: beta, safe serializer only.
- RESP server: experimental, loopback-first, bounded, and authenticated when
  bound outside loopback.
- LangChain helpers: optional integration, not part of the NSOS MVP release
  gate.

## Safe Server Usage

Default bind is loopback:

```bash
cargo run --manifest-path modules/oxtamem/oxta_engine/Cargo.toml --bin oxta_mem_server -- \
  --host 127.0.0.1 \
  --port 6379 \
  --db-path geodesic.db \
  --size-mb 100
```

Binding outside loopback requires an auth token:

```bash
OXTAMEM_AUTH_TOKEN=change-me cargo run --manifest-path modules/oxtamem/oxta_engine/Cargo.toml --bin oxta_mem_server -- \
  --host 0.0.0.0 \
  --port 6379
```

## Python SDK Serialization

The SDK intentionally does not use pickle. Supported values:

- `bytes`
- `str`
- JSON-compatible values
- NumPy arrays without object dtype
- PyTorch tensors, stored through a safe NumPy tensor payload

Unsupported Python objects raise `TypeError`.

## Validation

```bash
cargo test --manifest-path modules/oxtamem/oxta_engine/Cargo.toml --all-targets
cargo clippy --manifest-path modules/oxtamem/oxta_engine/Cargo.toml --all-targets -- -D warnings
```
