#!/usr/bin/env python3
# OXN/scripts/verify_determinism.py - Verificacao de consistencia de seeds

import argparse
import hashlib
import os
import sys
from pathlib import Path

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


def _import_nsos_ext(explicit_build_dir: str | None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nsos" / "scripts"))
    from native_module import load_native_module, resolve_native_build_dir
    explicit = explicit_build_dir or os.environ.get("NSOS_BUILD_DIR")
    directory = resolve_native_build_dir(Path(explicit) if explicit else None,
                                         _candidate_build_paths(None))
    module = load_native_module(directory)
    print(f"Native artifact: {module.__file__}; sha256={hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}")
    return module


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
