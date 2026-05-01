#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = REPO_ROOT / "PROJECT_BOUNDARY.json"


def _as_paths(values: Iterable[str]) -> list[Path]:
    return [(REPO_ROOT / value).resolve() for value in values]


def _load_boundary() -> dict:
    try:
        with BOUNDARY_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"failed to load {BOUNDARY_PATH}: {exc}") from exc


def _collect_declared_paths(boundary: dict) -> list[tuple[str, Path]]:
    declared: list[tuple[str, Path]] = []

    for path in _as_paths(boundary.get("supported_product", {}).get("paths", [])):
        declared.append(("supported_product", path))

    for module in boundary.get("integrated_modules", []):
        for path in _as_paths(module.get("paths", [])):
            declared.append((f"integrated_module:{module.get('name', 'unknown')}", path))

    for path in _as_paths(boundary.get("release_support", {}).get("paths", [])):
        declared.append(("release_support", path))

    for path in _as_paths(boundary.get("incubation_roots", [])):
        declared.append(("incubation", path))

    return declared


def _validate_required_sections(boundary: dict) -> list[str]:
    errors: list[str] = []
    required_sections = [
        "policy",
        "supported_product",
        "integrated_modules",
        "release_support",
        "incubation_roots",
        "promotion_requirements",
    ]
    for section in required_sections:
        if section not in boundary:
            errors.append(f"missing section: {section}")

    product_paths = boundary.get("supported_product", {}).get("paths", [])
    if "OXN/nsos" not in product_paths:
        errors.append("supported_product must include OXN/nsos")

    integrated_paths = [
        path
        for module in boundary.get("integrated_modules", [])
        for path in module.get("paths", [])
    ]
    if "modules/oxtamem" not in integrated_paths:
        errors.append("integrated_modules must include modules/oxtamem")

    if boundary.get("policy", {}).get("default_status") != "incubation":
        errors.append("policy.default_status must be incubation")

    return errors


def main() -> int:
    boundary = _load_boundary()
    errors = _validate_required_sections(boundary)

    declared = _collect_declared_paths(boundary)
    for label, path in declared:
        if not path.exists():
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            errors.append(f"{label} path does not exist: {rel}")

    seen: dict[Path, str] = {}
    for label, path in declared:
        previous = seen.get(path)
        if previous and previous != label:
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            errors.append(f"path declared in multiple boundary classes: {rel} ({previous}, {label})")
        seen[path] = label

    if errors:
        for error in errors:
            print(f"[boundary:error] {error}", file=sys.stderr)
        return 1

    print(
        "[boundary:ok] product=OXN/nsos integrated=modules/oxtamem "
        f"declared_paths={len(declared)} default=incubation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
