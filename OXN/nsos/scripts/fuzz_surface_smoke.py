from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight fuzz smoke tests for NSOS runtime surfaces.")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def detect_build_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build-mvp", "build_cuda129", "build_v1", "build_full", "build_codex", "build-ci-local", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("nsos_ext*.pyd")):
            return candidate.resolve()
    raise RuntimeError("Could not find a build directory with nsos_ext.")


def load_nsos(build_dir: Path):
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    import nsos_ext as nsos  # type: ignore

    return nsos


def build_config(nsos):
    config = nsos.ModelConfig()
    config.num_layers = 2
    config.d_model = 64
    config.vocab_size = 320
    config.n_heads = 4
    config.n_kv_heads = 2
    config.attention_period = 64
    config.attention_slot = 63
    config.use_moe = False
    config.use_ttt = False
    config.use_exact_attention_training = False
    config.max_context_tokens = 128
    return config


def find_api_binary(build_dir: Path) -> Path | None:
    names = ["nsos_api_server.exe", "nsos_api_server"]
    for name in names:
        candidate = build_dir / name
        if candidate.exists():
            return candidate
    return None


def send_raw_http(port: int, payload: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)
    config = build_config(nsos)
    auth_token = os.environ.get("NSOS_API_TOKEN") or "nsos-fuzz-smoke-token"

    report = {
        "build_dir": str(build_dir),
        "tokenizer_corruption": {"ok": False},
        "checkpoint_corruption": {"ok": False},
        "http_malformed_requests": {"ok": False, "skipped": False},
    }

    with tempfile.TemporaryDirectory(prefix="nsos-fuzz-") as tmp:
        tmpdir = Path(tmp)
        model_path = tmpdir / "model.bin"
        tokenizer_path = tmpdir / "tokenizer.nsos"
        corrupt_model_path = tmpdir / "model_corrupt.bin"
        corrupt_tokenizer_path = tmpdir / "tokenizer_corrupt.nsos"

        tokenizer = nsos.Tokenizer()
        tokenizer.add_special_tokens(["<|endoftext|>"])
        tokenizer.save_pack(str(tokenizer_path))

        model = nsos.JambaModel(config, nsos.Device.CPU)
        model.save(str(model_path))

        token_bytes = tokenizer_path.read_bytes()
        corrupt_tokenizer_path.write_bytes(token_bytes[: max(4, len(token_bytes) // 3)])
        try:
            bad_tokenizer = nsos.Tokenizer()
            bad_tokenizer.load(str(corrupt_tokenizer_path))
        except Exception as exc:
            report["tokenizer_corruption"] = {"ok": True, "error": str(exc)}
        else:
            raise RuntimeError("Corrupted tokenizer pack unexpectedly loaded without error")

        model_bytes = model_path.read_bytes()
        corrupt_model_path.write_bytes(model_bytes[: max(16, len(model_bytes) // 2)])
        try:
            bad_model = nsos.JambaModel(config, nsos.Device.CPU)
            bad_model.load(str(corrupt_model_path), False)
        except Exception as exc:
            report["checkpoint_corruption"] = {"ok": True, "error": str(exc)}
        else:
            raise RuntimeError("Corrupted checkpoint unexpectedly loaded without error")

        api_binary = find_api_binary(build_dir)
        if api_binary is None:
            report["http_malformed_requests"] = {"ok": True, "skipped": True, "reason": "api binary missing"}
        else:
            port = random.randint(18080, 18999)
            env = os.environ.copy()
            env["NSOS_API_TOKEN"] = auth_token
            proc = subprocess.Popen(
                [
                    str(api_binary),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--layers",
                    "2",
                    "--d-model",
                    "64",
                    "--vocab",
                    "320",
                    "--auth-token",
                    auth_token,
                ],
                cwd=str(build_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                deadline = time.time() + 15.0
                started = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        break
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                            started = True
                            break
                    except OSError:
                        time.sleep(0.2)
                if not started:
                    stdout, stderr = proc.communicate(timeout=2)
                    raise RuntimeError(f"HTTP server failed to start for fuzz smoke. stdout={stdout} stderr={stderr}")

                responses = []
                auth_header = f"Authorization: Bearer {auth_token}\r\n".encode("ascii")
                responses.append(
                    send_raw_http(
                        port,
                        b"BOGUS / HTTP/1.1\r\nHost: 127.0.0.1\r\n" + auth_header + b"\r\n",
                    )
                )
                responses.append(
                    send_raw_http(
                        port,
                        b"POST /generate HTTP/1.1\r\n"
                        b"Host: 127.0.0.1\r\n"
                        + auth_header
                        + b"Content-Type: application/json\r\n"
                        b"Content-Length: 10\r\n\r\n{\"prompt\":",
                    )
                )
                responses.append(
                    send_raw_http(
                        port,
                        b"GET /metrics HTTP/1.1\r\n"
                        b"Host: 127.0.0.1\r\n"
                        + auth_header
                        + b"X-Weird: \x01\x02\x03\r\n\r\n",
                    )
                )
                if proc.poll() is not None:
                    raise RuntimeError("HTTP server crashed after malformed requests")
                report["http_malformed_requests"] = {
                    "ok": True,
                    "skipped": False,
                    "response_sizes": [len(chunk) for chunk in responses],
                }
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()

    report_path = args.report_path or (Path(build_dir) / "fuzz_surface_smoke.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"[done] fuzz smoke report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
