#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_datasets.py — fetch public datasets for Oxta Contábil training.

Pipeline target (see comments per dataset):
  1. CulturaX PT-BR subset      → pre-training continuation (40M → 1B scale)
  2. BR-TaxQA-R                 → SFT gold (478 real tax Q&A + CARF rulings)
  3. BACEN FAQ                  → SFT (~2k financial Q&A from Central Bank BR)
  4. tech4humans/br-doc-extr    → vision/extraction (1220 BR docs incl. NF-e)
  5. LeNER-Br                   → NER (legal, reused for Oxta Jurídico later)

Output layout (relative to --output-dir):
  oxta_contabil/
    raw/
      culturax_ptbr/        # streamed .parquet shards
      br_taxqa/             # questions_QA_2024_v1.0.json + refs
      bacen_faq/            # q/a pairs
      br_doc_extraction/    # images + json schemas
      lener_br/             # CoNLL-style annotated text
    manifest.json           # what got fetched, sizes, sha256, license
    LICENSES.md             # license per dataset, attribution

Honest scope:
  - CulturaX is BIG (~300B tokens PT-BR full). Default fetches a sampled
    subset (~2-5GB) suitable to continue training a 400M-1B model.
  - For 7B+ scale you'd want the full set; pass --target-bytes 50GB.
  - All datasets here are commercial-OK (CC-BY-4.0, ODC-BY 1.0, CC0,
    domínio público) EXCEPT tech4humans/br-doc-extraction which has no
    declared license — we warn but download anyway. Verify before
    commercial use.

Usage:
  python download_datasets.py --output-dir ./artifacts/data
  python download_datasets.py --include br_taxqa,bacen_faq  # subset only
  python download_datasets.py --target-bytes 5GB --culturax-shards 20

Requires:
  pip install datasets huggingface_hub tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 on stdout/stderr — Windows console defaults to cp1252 which
# chokes on any non-Latin-1 char (em-dash, arrows, box drawing, etc.).
# Without this, printing "─" or "→" raises UnicodeEncodeError and kills
# the run mid-download.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Soft import: fail fast with a clear message if deps are missing ──
try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:
    sys.stderr.write(
        "ERROR: missing dependency. Install with:\n"
        "  pip install 'datasets>=2.16' huggingface_hub tqdm\n"
        f"\n(import error was: {exc})\n"
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# Dataset registry — single source of truth for what we fetch + how
# ─────────────────────────────────────────────────────────────────────

DATASETS = {
    "culturax_ptbr": {
        "hf_id":         "uonlp/CulturaX",
        "config":        "pt",
        "split":         "train",
        "license":       "ODC-BY-1.0",
        "commercial":    True,
        "purpose":       "Pre-training continuation in Brazilian Portuguese",
        "size_full":     "~300GB compressed (full pt subset)",
        "streaming":     True,
        "default_bytes": 5 * 1024 * 1024 * 1024,  # 5GB default sample
        "filter_keywords": [
            # Up-sample fiscal/accounting content during streaming
            "nota fiscal", "nf-e", "nfe", "cfop", "ncm", "icms", "ipi",
            "iss", "pis", "cofins", "irpj", "csll", "sped", "ecf", "efd",
            "contador", "contábil", "contábeis", "tributo", "tributário",
            "balanço patrimonial", "dre", "lucro líquido", "receita federal",
            "simples nacional", "lucro real", "lucro presumido",
        ],
    },

    "br_taxqa": {
        "hf_id":         "unicamp-dl/BR-TaxQA-R",
        "config":        None,
        "split":         None,
        "license":       "CC-BY-4.0",
        "commercial":    True,
        "purpose":       "SFT gold — 478 IRPF Q&A with CARF case law",
        "size_full":     "~10MB",
        "streaming":     False,
        "snapshot":      True,  # full repo snapshot (JSON files)
    },

    "bacen_faq": {
        # Best-effort — community FAQ datasets exist under slightly different
        # IDs. We try a primary then fall back; if both fail we leave a note.
        "hf_id":         "celsowm/bacen_faqs_qa",
        "fallback_ids":  ["TucanoBR/bacen-faqs", "rufimelo/bacen-faq"],
        "config":        None,
        "split":         "train",
        "license":       "Domínio público (BACEN é orgão público)",
        "commercial":    True,
        "purpose":       "SFT — financial vocabulary, ~2k Q&A pairs",
        "size_full":     "~2MB",
        "streaming":     False,
        "snapshot":      False,
    },

    "br_doc_extraction": {
        "hf_id":         "tech4humans/br-doc-extraction",
        "config":        None,
        "split":         None,
        "license":       "UNDECLARED — verify before commercial use",
        "commercial":    None,
        "purpose":       "Vision: 1220 BR docs (CNH, RG, NF) + JSON schemas",
        "size_full":     "~500MB",
        "streaming":     False,
        "snapshot":      True,
    },

    "lener_br": {
        # Brazilian legal NER. Primary route is the academic mirror;
        # community uploads exist on HF under varied IDs.
        "hf_id":         "peluz/lener_br",
        "fallback_ids":  ["lener_br", "neuralmind/lener_br"],
        "config":        None,
        "split":         "train",
        "license":       "CC-BY-4.0",
        "commercial":    True,
        "purpose":       "NER — Brazilian legal entities (Oxta Jurídico seed)",
        "size_full":     "~30MB",
        "streaming":     False,
        "snapshot":      False,
    },
}

ALL_NAMES = list(DATASETS.keys())


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def parse_bytes(spec: str) -> int:
    """Accept '5GB', '500MB', '1.5GB', or plain integer (bytes)."""
    spec = spec.strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB|B)?", spec)
    if not m:
        raise ValueError(f"Unrecognized byte spec: {spec!r}")
    value = float(m.group(1))
    unit = m.group(2) or "B"
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    return int(value * mult)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file. Handles multi-GB files."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def directory_total_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def directory_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def keyword_match(text: str, keywords: List[str]) -> bool:
    """Case-insensitive contains-any check.  Fast path: lowercase once."""
    lower = text.lower()
    return any(k in lower for k in keywords)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────
# Per-dataset fetchers
# ─────────────────────────────────────────────────────────────────────

def fetch_culturax(name: str, info: Dict[str, Any], dst_root: Path,
                   target_bytes: int, max_shards: int,
                   filter_fiscal: bool) -> Dict[str, Any]:
    """
    CulturaX is huge — we stream a sampled subset and write parquet shards
    of fiscal-enriched text.  Stops when total written bytes >= target.
    """
    out = dst_root / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] streaming {info['hf_id']} (config={info['config']!r}) …")
    print(f"[{name}] target ≈ {human_bytes(target_bytes)}, "
          f"keyword up-sample={'on' if filter_fiscal else 'off'}")

    try:
        ds = load_dataset(
            info["hf_id"], info["config"],
            split=info["split"], streaming=True,
        )
    except Exception as exc:
        return {"status": "error", "error": f"load_dataset failed: {exc}"}

    written_bytes = 0
    written_docs = 0
    fiscal_docs = 0
    shard_idx = 0
    shard_lines: List[str] = []
    shard_target_bytes = max(64 * 1024 * 1024, target_bytes // max_shards)
    shard_bytes = 0

    keywords = info.get("filter_keywords", []) if filter_fiscal else []

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_lines
        if not shard_lines:
            return
        shard_path = out / f"culturax_ptbr_shard_{shard_idx:04d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            f.writelines(shard_lines)
        print(f"[{name}]   wrote shard {shard_idx:04d} "
              f"({len(shard_lines)} docs, {human_bytes(shard_bytes)})")
        shard_idx += 1
        shard_lines = []
        shard_bytes = 0

    try:
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 200:
                continue

            is_fiscal = bool(keywords) and keyword_match(text, keywords)
            # Keep all fiscal docs; sample 10% of generic ones to stay broad.
            if not is_fiscal and (written_docs % 10) != 0:
                # tracked but not written
                continue

            line = json.dumps(
                {"text": text, "is_fiscal": is_fiscal},
                ensure_ascii=False,
            ) + "\n"
            shard_lines.append(line)
            line_bytes = len(line.encode("utf-8"))
            shard_bytes += line_bytes
            written_bytes += line_bytes
            written_docs += 1
            if is_fiscal:
                fiscal_docs += 1

            if shard_bytes >= shard_target_bytes:
                flush_shard()
                if shard_idx >= max_shards:
                    print(f"[{name}]   reached max-shards={max_shards}, stopping")
                    break

            if written_bytes >= target_bytes:
                print(f"[{name}]   reached target {human_bytes(target_bytes)}, stopping")
                break

    except KeyboardInterrupt:
        print(f"\n[{name}]   interrupted by user, flushing partial shard …")
    finally:
        flush_shard()

    return {
        "status":       "ok",
        "bytes":        written_bytes,
        "docs":         written_docs,
        "fiscal_docs":  fiscal_docs,
        "shards":       shard_idx,
        "filter_applied": filter_fiscal,
    }


def fetch_snapshot(name: str, info: Dict[str, Any],
                   dst_root: Path) -> Dict[str, Any]:
    """For datasets with a few small files — pull the whole repo snapshot."""
    out = dst_root / name
    out.mkdir(parents=True, exist_ok=True)
    candidates = [info["hf_id"]] + list(info.get("fallback_ids", []))

    for candidate in candidates:
        print(f"[{name}] snapshot_download({candidate!r}) → {out} …")
        try:
            snapshot_download(
                repo_id=candidate,
                repo_type="dataset",
                local_dir=str(out),
                local_dir_use_symlinks=False,
                # Some configs ship parquet shards; we want everything small.
            )
            return {
                "status":   "ok",
                "hf_id":    candidate,
                "bytes":    directory_total_bytes(out),
                "files":    directory_file_count(out),
            }
        except HfHubHTTPError as exc:
            print(f"[{name}]   {candidate} failed ({exc.response.status_code}); "
                  f"trying next fallback …")
        except Exception as exc:
            print(f"[{name}]   {candidate} failed: {exc}")

    return {
        "status": "error",
        "error": f"all candidates failed: {candidates}",
    }


def fetch_load_dataset(name: str, info: Dict[str, Any],
                       dst_root: Path) -> Dict[str, Any]:
    """Standard load_dataset + save_to_disk path for small structured sets."""
    out = dst_root / name
    out.mkdir(parents=True, exist_ok=True)
    candidates = [info["hf_id"]] + list(info.get("fallback_ids", []))

    for candidate in candidates:
        print(f"[{name}] load_dataset({candidate!r}, split={info.get('split')!r}) …")
        try:
            kwargs = {}
            if info.get("config"):
                kwargs["name"] = info["config"]
            if info.get("split"):
                kwargs["split"] = info["split"]
            ds = load_dataset(candidate, **kwargs)
            ds.save_to_disk(str(out))
            return {
                "status":   "ok",
                "hf_id":    candidate,
                "bytes":    directory_total_bytes(out),
                "rows":     len(ds) if hasattr(ds, "__len__") else None,
            }
        except Exception as exc:
            print(f"[{name}]   {candidate} failed: {exc}")

    return {
        "status": "error",
        "error": f"all candidates failed: {candidates}",
    }


# ─────────────────────────────────────────────────────────────────────
# License file writer — single source of truth for attribution
# ─────────────────────────────────────────────────────────────────────

def write_licenses_file(dst_root: Path, results: Dict[str, Dict[str, Any]]) -> None:
    """
    Renders LICENSES.md with one section per dataset.  Used for legal
    review before any commercial deployment and for attribution credit
    in the released product / blog post.
    """
    lines = ["# Licenses & Attribution\n",
             f"_Generated {now_iso()} by download_datasets.py_\n",
             "\nUse this file when checking whether Oxta Contábil may "
             "ship trained on these inputs commercially, and when "
             "writing the credits/attribution section of the public "
             "release.\n"]

    for name in ALL_NAMES:
        info = DATASETS[name]
        result = results.get(name, {})
        lines.append(f"\n## {name}\n")
        lines.append(f"- **Source:** `{info['hf_id']}` (HuggingFace)\n")
        lines.append(f"- **License:** {info['license']}\n")
        if info["commercial"] is True:
            lines.append("- **Commercial use:** OK with attribution\n")
        elif info["commercial"] is False:
            lines.append("- **Commercial use:** ❌ NOT allowed\n")
        else:
            lines.append("- **Commercial use:** ⚠️ verify before shipping\n")
        lines.append(f"- **Purpose:** {info['purpose']}\n")
        if result.get("status") == "ok":
            lines.append(f"- **Fetched:** {human_bytes(result.get('bytes', 0))}\n")
        elif result.get("status") == "error":
            lines.append(f"- **Fetch status:** ⚠️ failed — {result.get('error')}\n")

    (dst_root / "LICENSES.md").write_text(
        "".join(lines), encoding="utf-8"
    )
    print(f"\nWrote {dst_root / 'LICENSES.md'}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download public datasets for Oxta Contábil training.",
    )
    p.add_argument(
        "--output-dir",
        default="./artifacts/data",
        help="Where to write oxta_contabil/* trees (default: ./artifacts/data)",
    )
    p.add_argument(
        "--include",
        default="all",
        help=f"Comma-separated names to fetch.  Available: {','.join(ALL_NAMES)} "
             f"(default: all)",
    )
    p.add_argument(
        "--target-bytes",
        default="5GB",
        help="Stop CulturaX streaming at ≈ this many bytes (e.g. 2GB, 50GB). "
             "Other datasets are downloaded in full.",
    )
    p.add_argument(
        "--culturax-shards",
        type=int, default=64,
        help="Max parquet shards to write for CulturaX (default 64).",
    )
    p.add_argument(
        "--no-fiscal-filter",
        action="store_true",
        help="Disable fiscal-keyword up-sampling on CulturaX.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched but don't download.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    dst_root = output_dir / "oxta_contabil" / "raw"
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    selected = ALL_NAMES if args.include == "all" else [
        n.strip() for n in args.include.split(",") if n.strip()
    ]
    unknown = [n for n in selected if n not in DATASETS]
    if unknown:
        sys.stderr.write(f"ERROR: unknown dataset(s): {unknown}\n")
        sys.stderr.write(f"Known: {ALL_NAMES}\n")
        return 1

    target_bytes = parse_bytes(args.target_bytes)

    print(f"output dir:       {output_dir}")
    print(f"target subdir:    {dst_root}")
    print(f"datasets:         {selected}")
    print(f"culturax target:  {human_bytes(target_bytes)}, "
          f"max-shards={args.culturax_shards}, "
          f"fiscal-filter={'off' if args.no_fiscal_filter else 'on'}")
    print(f"dry-run:          {args.dry_run}")
    print()

    if args.dry_run:
        for name in selected:
            print(f"  would fetch: {name} ({DATASETS[name]['hf_id']})")
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    started = time.time()

    for name in selected:
        info = DATASETS[name]
        print(f"\n{'=' * 60}\n {name}: {info['purpose']}\n{'=' * 60}")
        t0 = time.time()
        try:
            if name == "culturax_ptbr":
                r = fetch_culturax(
                    name, info, dst_root,
                    target_bytes=target_bytes,
                    max_shards=args.culturax_shards,
                    filter_fiscal=not args.no_fiscal_filter,
                )
            elif info.get("snapshot"):
                r = fetch_snapshot(name, info, dst_root)
            else:
                r = fetch_load_dataset(name, info, dst_root)
        except Exception as exc:
            r = {"status": "error", "error": str(exc)}

        r["elapsed_seconds"] = round(time.time() - t0, 2)
        r["fetched_at"] = now_iso()
        results[name] = r

        if r["status"] == "ok":
            print(f"[{name}] DONE in {r['elapsed_seconds']}s "
                  f"({human_bytes(r.get('bytes', 0))})")
        else:
            print(f"[{name}] FAILED: {r.get('error')}")

    # ── Manifest + LICENSES.md ──────────────────────────────────────
    manifest = {
        "generated_at":   now_iso(),
        "output_dir":     str(output_dir),
        "selected":       selected,
        "target_bytes":   target_bytes,
        "fiscal_filter":  not args.no_fiscal_filter,
        "total_seconds":  round(time.time() - started, 2),
        "results":        results,
    }
    (dst_root.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_licenses_file(dst_root.parent, results)

    # Final summary
    print("\n" + "═" * 60)
    print(" Summary")
    print("═" * 60)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    failed = sum(1 for r in results.values() if r["status"] != "ok")
    total_bytes = sum(r.get("bytes", 0) for r in results.values())
    print(f"  datasets ok:     {ok}/{len(selected)}")
    print(f"  total bytes:     {human_bytes(total_bytes)}")
    print(f"  total time:      {manifest['total_seconds']}s")
    print(f"  manifest:        {dst_root.parent / 'manifest.json'}")
    print(f"  licenses:        {dst_root.parent / 'LICENSES.md'}")
    if failed:
        print(f"\n  ⚠️ {failed} failed — check manifest.json for details")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
