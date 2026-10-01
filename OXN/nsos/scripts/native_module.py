"""Locate the native module using this interpreter's extension ABI suffixes."""
from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
import importlib.util
import hashlib
from pathlib import Path
import sys
from typing import Iterable


def has_native_module(directory: Path, name: str = "nsos_ext") -> bool:
    return directory.is_dir() and any(
        (directory / (name + suffix)).is_file()
        for suffix in EXTENSION_SUFFIXES
    )


def resolve_native_build_dir(explicit: Path | None, candidates: Iterable[Path] = ()) -> Path:
    """An explicit build is authoritative: never validate a fallback artifact."""
    roots = [explicit] if explicit is not None else list(candidates)
    for root in roots:
        root = Path(root).expanduser().resolve()
        for directory in (root, root / "Release"):
            if has_native_module(directory):
                return directory
    label = str(explicit) if explicit is not None else "automatic candidates"
    raise RuntimeError(f"No compatible nsos_ext native module in {label}")


def load_native_module(build_dir: Path):
    """Load the selected ABI file directly, independent of sys.path ordering."""
    directory = resolve_native_build_dir(build_dir)
    binary = next(directory / ("nsos_ext" + suffix) for suffix in EXTENSION_SUFFIXES
                  if (directory / ("nsos_ext" + suffix)).is_file())
    cached = sys.modules.get("nsos_ext")
    if cached is not None:
        origin = getattr(cached, "__file__", None)
        if origin is None or Path(origin).resolve() != binary.resolve():
            raise RuntimeError("nsos_ext already loaded from a different build; use a fresh process")
        return cached
    from cuda_env import add_windows_runtime_dirs
    add_windows_runtime_dirs(directory)
    spec = importlib.util.spec_from_file_location("nsos_ext", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native module: {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nsos_ext"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get("nsos_ext") is module:
            del sys.modules["nsos_ext"]
        raise
    return module


def native_artifact_identity(module) -> dict[str, str]:
    binary = Path(module.__file__).resolve()
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(binary), "sha256": digest.hexdigest()}
