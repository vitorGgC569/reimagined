"""Container startup and readiness using the HTTP server's security contract."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import sys
import urllib.request


def server_arguments(env) -> list[str]:
    token = env.get("NSOS_API_TOKEN", "")
    if len(token.encode("utf-8")) < 16:
        raise ValueError("NSOS_API_TOKEN must contain at least 16 bytes")
    proxies = [ip.strip() for ip in env.get("NSOS_TRUSTED_PROXY_IPS", "").split(",") if ip.strip()]
    if not proxies:
        raise ValueError("NSOS_TRUSTED_PROXY_IPS must name the TLS proxy peers")
    for peer in proxies:
        ipaddress.ip_address(peer)
    port = int(env.get("NSOS_PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("NSOS_PORT must be in 1..65535")
    model = env.get("NSOS_MODEL", "")
    if not model or not Path(model).exists():
        raise ValueError("NSOS_MODEL must identify a mounted model pack or checkpoint")
    return [
        "/opt/nsos/bin/nsos_api_server", "--host", "0.0.0.0", "--port", str(port),
        "--model", model, "--trust-proxy-headers", "--require-tls-proxy-header",
        "--trusted-proxy-ips", ",".join(proxies),
        "--pack-root", env.get("NSOS_PACK_ROOT", "/opt/nsos/artifacts/model_packs"),
    ]


def check_ready(env) -> None:
    port = int(env.get("NSOS_PORT", "8080"))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/ready",
        headers={"Authorization": "Bearer " + env.get("NSOS_API_TOKEN", "")},
    )
    # Never send a health probe (and its token) through an environment proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=3) as response:
        payload = json.loads(response.read(65536))
        if response.status != 200 or payload.get("status") != "ready" or payload.get("ok") is not True:
            raise RuntimeError("NSOS has not initialized its model and inference replicas")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv == ["health"]:
            check_ready(os.environ)
        elif not argv or argv == ["serve"]:
            arguments = server_arguments(os.environ)
            os.execv(arguments[0], arguments)
        else:
            raise ValueError("Expected serve or health")
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[container] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
