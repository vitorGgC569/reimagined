#!/usr/bin/env python3
# OXN/scripts/verify_determinism.py - Verificacao de consistencia de seeds

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

_DLL_HANDLES = []


def _candidate_build_paths(explicit_build_dir: str | None) -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    oxn_root = script_dir.parent
    nsos_root = oxn_root / "nsos"
    candidates = []
    if explicit_build_dir:
        candidates.append(Path(explicit_build_dir))
    env_build_dir = os.environ.get("NSOS_BUILD_DIR")
    if env_build_dir:
        candidates.append(Path(env_build_dir))
    candidates.extend(
        [
            nsos_root / "build-ci",
            nsos_root / "build_cuda129",
            nsos_root / "build",
        ]
    )
    return candidates


def _bootstrap_build_path(explicit_build_dir: str | None) -> None:
    for candidate in _candidate_build_paths(explicit_build_dir):
        candidate = candidate.resolve()
        for path in (candidate, candidate / "Release"):
            if path.exists():
                if hasattr(os, "add_dll_directory"):
                    _DLL_HANDLES.append(os.add_dll_directory(str(path)))
                sys.path.insert(0, str(path))


def _import_nsos_ext(explicit_build_dir: str | None):
    candidates = _candidate_build_paths(explicit_build_dir)
    _bootstrap_build_path(explicit_build_dir)
    try:
        import nsos_ext  # type: ignore
        return nsos_ext
    except ImportError as exc:
        for candidate in candidates:
            candidate = candidate.resolve()
            search_roots = [candidate, candidate / "Release"]
            for root in search_roots:
                if not root.exists():
                    continue
                matches = sorted(root.glob("nsos_ext*.pyd"))
                if not matches:
                    continue
                spec = importlib.util.spec_from_file_location("nsos_ext", matches[0])
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["nsos_ext"] = module
                    spec.loader.exec_module(module)
                    return module
        print(f"Falha ao importar nsos_ext. Verifique se o build foi gerado. Detalhe: {exc}")
        raise SystemExit(1) from exc


def _hash_result(result):
    if hasattr(result, "numpy"):
        arr = result.numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()
    return hashlib.sha256(str(result).encode()).hexdigest()


def verify_determinism(nsos_ext) -> None:
    print("--- Verificando Determinismo do NSOS ---")

    seeds = [42, 123, 999]
    all_passed = True

    for seed in seeds:
        print(f"Testando seed: {seed}...", end=" ")

        nsos_ext.set_seed(seed)
        first = nsos_ext.Tensor.random([10, 10], nsos_ext.Device.CPU)
        first_hash = _hash_result(first)

        nsos_ext.set_seed(seed)
        second = nsos_ext.Tensor.random([10, 10], nsos_ext.Device.CPU)
        second_hash = _hash_result(second)

        if first_hash == second_hash:
            print("OK")
        else:
            print("FALHA")
            all_passed = False

    if all_passed:
        print("\nTodos os testes de determinismo passaram.")
    else:
        print("\nProblemas de determinismo detectados.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=None)
    args = parser.parse_args()
    nsos_ext = _import_nsos_ext(args.build_dir)
    verify_determinism(nsos_ext)
