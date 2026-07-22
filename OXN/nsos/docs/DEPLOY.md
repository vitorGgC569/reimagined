# Deployment Guidance

`nsos_api_server` is the only supported HTTP serving binary. The retired
`src/server.cpp` demonstration is intentionally non-operational.

## Required security posture

- Token authentication is mandatory. Use `NSOS_API_TOKEN` or `--auth-token`.
  The only exception is explicit `--allow-unauthenticated-local` on loopback.
- Non-loopback binding is accepted only behind a TLS-terminating proxy and
  requires all of `--trust-proxy-headers`, `--require-tls-proxy-header`, and
  `--trusted-proxy-ips <ip[,ip...]>`. Forwarded headers from any other peer are
  ignored and cannot assert HTTPS or choose a rate-limit identity.
- Use an authentication token of at least 16 bytes outside loopback.
- Administrative training and pack endpoints are disabled unless
  `--enable-admin-endpoints` is supplied. Keep them disabled on public serving
  processes.
- Wildcard CORS is opt-in through `--allow-cors` and is intended only for local
  or separately protected deployments.

Example behind a reverse proxy on `10.0.0.10`:

```sh
nsos_api_server --model /models/pack \
  --host 0.0.0.0 --port 8080 \
  --auth-token "$NSOS_API_TOKEN" \
  --trust-proxy-headers \
  --require-tls-proxy-header \
  --trusted-proxy-ips 10.0.0.10
```

The proxy must overwrite, not append, `X-Forwarded-For` and
`X-Forwarded-Proto`, and must send `X-Forwarded-Proto: https`.

## Health and readiness

- `GET /health` reports process liveness (`status=alive`).
- `GET /ready` returns HTTP 200 only when workers, the loaded model, and the
  inference replica pool are initialized. It returns HTTP 503 otherwise.
- `GET /metrics` requires authentication unless the deployment deliberately
  changes the health authentication policy.

## Resource controls

Keep finite bounds for headers, bodies, worker queue, socket timeout,
generation tokens, batch prompts, training sizes, rate-limit clients, and SSE
pending bytes. Relevant CLI controls include:

- `--max-header-bytes`, `--max-body-bytes`, `--max-queue-depth`
- `--rate-limit-rpm`, `--max-rate-limit-clients`
- `--max-stream-pending-bytes`, `--socket-timeout-ms`
- `--max-generate-tokens`, `--max-request-context`, `--max-batch-prompts`
- the `--max-train-*` family

SSE generation applies bounded backpressure. A disconnected or stalled client
cannot grow an unbounded token queue.

## Model packs

Set `--pack-root` to a dedicated directory. Relative pack targets are
canonicalized beneath it; path traversal and symlink escape are rejected.
Absolute output paths remain disabled unless explicitly enabled. Pack children
are durably flushed and atomically replaced, with the checksummed manifest
published last.

## OxtaMem

The companion service is loopback-first and requires authentication outside
loopback. Its durable mode uses a checksummed incremental metadata journal and
periodic full snapshots. `set_sync_on_write(false)` is only for bulk ingestion
that accepts loss of writes since the last explicit `flush()` after a crash.

## Local-only example

```sh
nsos_api_server --model /models/pack \
  --host 127.0.0.1 --port 8080 \
  --auth-token "$NSOS_API_TOKEN"
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/info \
  -H "Authorization: Bearer $NSOS_API_TOKEN"
```
