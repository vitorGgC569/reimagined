# NSOS HTTP API

The supported HTTP API is served by `nsos_api_server`.

Authentication is required by default. Use `NSOS_API_TOKEN` or `--auth-token`.
Unauthenticated local mode is only for explicit loopback development with
`--allow-unauthenticated-local`.

## Start

```powershell
$env:NSOS_API_TOKEN = "change-me"
.\OXN\nsos\build-mvp\Release\nsos_api_server.exe `
  --host 127.0.0.1 `
  --port 8080 `
  --auth-token $env:NSOS_API_TOKEN
```

## Public Endpoints

- `GET /health`
- `GET /ready`
- `GET /`
- `GET /info`
- `GET /metrics`
- `POST /generate`
- `POST /generate-batch`
- `POST /generate-stream`

Authenticated requests use:

```text
Authorization: Bearer <token>
```

## Admin Endpoints

Admin endpoints are disabled unless the server starts with
`--enable-admin-endpoints`.

- `POST /train-text`
- `POST /train-batch`
- `POST /train-corpus`
- `POST /pack`

Model pack output paths are rooted under the configured pack root by default.
Absolute pack paths are disabled unless explicitly allowed.

## Operational Limits

The server enforces request body limits, header limits, queue limits, socket
timeouts, JSON depth limits, generation limits, and rate limiting. These are part
of the product surface and should remain covered by tests and fuzz smoke.
