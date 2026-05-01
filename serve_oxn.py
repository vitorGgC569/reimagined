"""Legacy entrypoint guard.

The previous implementation exposed an unauthenticated mock REST server on all
interfaces and generated dummy token IDs. It is intentionally disabled for the
MVP. Use OXN/nsos/build-mvp/Release/nsos_api_server.exe or the Linux equivalent
instead.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "serve_oxn.py is disabled. Start the official NSOS API server from "
        "OXN/nsos/build-mvp with --auth-token or NSOS_API_TOKEN.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
