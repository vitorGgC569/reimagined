# Deployment Guidance

> **Status:** **skeleton (Phase 0 PR-0.3)**. Full content (TLS guidance, healthcheck wiring, reverse proxy snippets, FFI failure handling) lands in **Phase 4**.
>
> Until Phase 4 lands, this file is the placeholder pointer; if you need to deploy today, follow the constraints listed under "Hard constraints" below — they apply already.

---

## Hard constraints (apply now, will not change)

These are the constraints the supported product currently enforces; deployments outside them are unsupported.

- **Authentication** — `nsos_api_server` enforces token auth when `--host` is not loopback. Required env var (or flag): `NSOS_API_TOKEN` / `--auth-token`. No bypass except `--allow-unauthenticated-local` and only when explicitly set.
- **Admin endpoints** — `/train-text`, `/train-batch`, `/train-corpus`, `/pack` are gated behind `--enable-admin-endpoints`. They must remain off in any deployment that exposes the port outside trusted infrastructure.
- **TLS** — TLS termination is the responsibility of a reverse proxy (nginx, caddy, cloud load balancer). The product reads `X-Forwarded-Proto` only when `--require-tls-proxy-header` is set. Running `nsos_api_server` directly on a public interface is **not supported**.
- **OxtaMem companion** — must remain loopback-bound (`127.0.0.1`) by default. Authentication required for non-loopback. Frame, connection, and timeout limits are documented in `modules/oxtamem/oxta_engine/src/main.rs` and must not be raised without a documented threat model.
- **Resource limits** — request body size, header size, queue depth, rate-limit, and per-request timeouts are configured in `OXN/nsos/src/api_server.cpp::ApiServerConfig`. Do not raise them silently; document any change.
- **Pack root** — when `nsos_api_server` is started with admin endpoints, set `--pack-root` to a directory whose canonicalized path stays inside a single configured root. The HTTP API rejects any pack target outside that root.

## To be authored in Phase 4

- **Healthcheck on HTTP** — Docker `HEALTHCHECK` invoking `curl -fsS http://127.0.0.1:${NSOS_PORT:-8080}/info -H "Authorization: Bearer ${NSOS_HEALTH_TOKEN}"`. The current Python-import-only healthcheck is to be replaced.
- **Minimum nginx and caddy snippets** — terminating TLS and forwarding `X-Forwarded-Proto` to `nsos_api_server`.
- **Reference hardware** — for `edge_throughput_tokens_per_second_cpu` measurement (Phase 6 scorecard threshold).
- **OxtaMem FFI failure modes** — what to expect in logs when `oxtamem_create()` / `oxtamem_open()` fails, and how to choose between abort vs `--no-oxtamem` degraded mode (logging will be added in Phase 4.D).
- **Build matrix** — supported OS / compiler combinations with checksums for the Dockerfile build context.
- **Secrets handling** — recommendations for injecting `NSOS_API_TOKEN`, `NSOS_HEALTH_TOKEN`, `OXTAMEM_AUTH_TOKEN` (env vars vs secret managers).
- **Operational runbook** — what to do when a release gate fails in production: rollback, log collection, and bisection.

## Today: minimal Docker run

```sh
docker build -t nsos-mvp .
docker run --rm -d --name nsos \
  -e NSOS_API_TOKEN=changeme \
  -p 127.0.0.1:8080:8080 \
  nsos-mvp
curl -fsS http://127.0.0.1:8080/info -H "Authorization: Bearer changeme"
```

Note: this maps to `127.0.0.1` only. Do not bind `0.0.0.0` until you have a reverse proxy in front and `--require-tls-proxy-header` set.
